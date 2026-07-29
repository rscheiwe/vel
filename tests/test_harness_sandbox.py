import pytest

from vel.harness.config import SandboxConfig
from vel.harness.sandbox import build_sandbox_tools
from vel.harness.sandbox.providers import E2BSandboxProvider, LocalSubprocessSandboxProvider


@pytest.mark.asyncio
async def test_sandbox_tools_mark_mutating_and_exec_for_confirmation(tmp_path):
    session = await _local_session(tmp_path)
    tools = build_sandbox_tools(
        session,
        ['read', 'write', 'edit', 'list', 'bash', 'python'],
        timeout_seconds=5,
    )

    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        'sandbox_read',
        'sandbox_write',
        'sandbox_edit',
        'sandbox_list',
        'sandbox_bash',
        'sandbox_python',
    }
    assert by_name['sandbox_read'].requires_confirmation is False
    assert by_name['sandbox_list'].requires_confirmation is False
    assert by_name['sandbox_write'].requires_confirmation is True
    assert by_name['sandbox_edit'].requires_confirmation is True
    assert by_name['sandbox_bash'].requires_confirmation is True
    assert by_name['sandbox_python'].requires_confirmation is True
    await session.close()


def test_sandbox_config_accepts_shipped_provider_name():
    config = SandboxConfig(provider='local_subprocess')
    assert config.provider == 'local_subprocess'


@pytest.mark.asyncio
async def test_local_sandbox_file_tools_use_workdir(tmp_path):
    session = await _local_session(tmp_path)
    tools = {tool.name: tool for tool in build_sandbox_tools(session, ['read', 'write', 'edit', 'list'])}

    await tools['sandbox_write']._handler(path='hello.txt', content='hello world')
    read_events = [event async for event in tools['sandbox_read']._handler(path='hello.txt')]
    read = read_events[-1]['output']
    assert read == {'path': 'hello.txt', 'content': 'hello world', 'encoding': 'utf-8'}

    edited = await tools['sandbox_edit']._handler(path='hello.txt', old='world', new='sandbox')
    assert edited['changed'] is True
    assert (tmp_path / 'workspace' / 'hello.txt').read_text() == 'hello sandbox'

    listed = await tools['sandbox_list']._handler(path='.')
    assert listed['entries'] == ['hello.txt']
    await session.close()


@pytest.mark.asyncio
async def test_sandbox_read_can_emit_file_event(tmp_path):
    session = await _local_session(tmp_path)
    tools = {tool.name: tool for tool in build_sandbox_tools(session, ['read', 'write'])}

    await tools['sandbox_write']._handler(path='report.md', content='hello browser')
    events = [
        event
        async for event in tools['sandbox_read']._handler(path='report.md', expose_file=True)
    ]

    assert events[0]['type'] == 'file'
    assert events[0]['name'] == 'report.md'
    assert events[0]['mimeType'] in ('text/markdown', 'text/x-markdown', 'application/octet-stream')
    assert events[0]['content']
    assert events[-1] == {
        'type': 'tool-output',
        'output': {'path': 'report.md', 'content': 'hello browser', 'encoding': 'utf-8'},
    }
    await session.close()


@pytest.mark.asyncio
async def test_local_sandbox_exec_tools(tmp_path):
    session = await _local_session(tmp_path)
    tools = {tool.name: tool for tool in build_sandbox_tools(session, ['bash', 'python'], timeout_seconds=5)}

    bash = await tools['sandbox_bash']._handler(cmd='printf ok')
    assert bash == {'stdout': 'ok', 'stderr': '', 'exit_code': 0}

    python = await tools['sandbox_python']._handler(code='print(2 + 3)')
    assert python == {'stdout': '5\n', 'stderr': '', 'exit_code': 0}
    await session.close()


@pytest.mark.asyncio
async def test_local_sandbox_rejects_implicit_unsafe_local(tmp_path):
    provider = LocalSubprocessSandboxProvider()
    with pytest.raises(ValueError, match='unsafe_local=True'):
        await provider.create(SandboxConfig(provider_options={'root': str(tmp_path)}))


@pytest.mark.asyncio
async def test_local_sandbox_blocks_path_escape(tmp_path):
    session = await _local_session(tmp_path)
    with pytest.raises(ValueError, match='escapes sandbox root'):
        await session.write_file('../../escape.txt', b'nope')
    await session.close()


@pytest.mark.asyncio
async def test_local_sandbox_connects_across_provider_instances(tmp_path):
    session = await _local_session(tmp_path)
    await session.write_file('state.txt', b'warm')

    provider = LocalSubprocessSandboxProvider()
    connected = await provider.connect(session.id)

    assert connected.id == session.id
    assert await connected.read_file('state.txt') == b'warm'
    await session.close()


@pytest.mark.asyncio
async def test_local_sandbox_reuses_explicit_root_for_warm_lifecycles(tmp_path):
    provider1 = LocalSubprocessSandboxProvider()
    provider2 = LocalSubprocessSandboxProvider()
    config = SandboxConfig(
        lifecycle='per_session',
        provider_options={'unsafe_local': True, 'root': str(tmp_path)},
    )

    session1 = await provider1.create(config)
    await session1.write_file('warm.txt', b'yes')
    session2 = await provider2.create(config)

    assert session2.id == session1.id
    assert await session2.read_file('warm.txt') == b'yes'
    await session1.close()


def _e2b_sdk_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec('e2b') is not None


@pytest.mark.asyncio
@pytest.mark.skipif(
    _e2b_sdk_installed(),
    reason='asserts the SDK-absent path; with e2b installed the call gets as far as auth',
)
async def test_e2b_provider_soft_imports_optional_sdk():
    """Without the optional extra, asking for an e2b sandbox must say so.

    The assertion is about the missing-extra message, so it is only meaningful
    when the SDK is genuinely absent — which is the case in CI, where only
    `[dev]` is installed. On a machine that has e2b the call reaches
    authentication instead and this would fail for a reason unrelated to what it
    is testing."""
    provider = E2BSandboxProvider()
    with pytest.raises(ImportError, match='sandbox-e2b'):
        await provider.create(SandboxConfig(provider='e2b'))


async def _local_session(tmp_path):
    provider = LocalSubprocessSandboxProvider()
    return await provider.create(
        SandboxConfig(
            workdir='/workspace',
            provider_options={'unsafe_local': True, 'root': str(tmp_path)},
        )
    )
