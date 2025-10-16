// run: OPENAI_API_KEY=... node scripts/ai_sdk_trace.mjs --prompt "Hello"
import { OpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";

const argv = yargs(hideBin(process.argv)).option("prompt",{type:"string",demandOption:true}).argv;

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const model = openai("gpt-4o"); // swap to responses model via options if desired

const norm = (part) => {
  const t = part.type; const out = { t };
  if (part.id) out.id = part.id;
  if (t === "text-delta" || t === "reasoning-delta") out.d = part.textDelta || part.delta || "";
  if (t === "tool-input-available") { out.tool = part.toolName; out.args = part.toolInput; }
  if (t === "tool-output-available") { out.tool = part.toolName; out.out = part.toolOutput; }
  return out;
};

const result = await streamText({
  model,
  prompt: argv.prompt,
  // For Responses API style + reasoning, you can switch provider/config here.
  // e.g., openai.responses({ reasoning: { summary: "auto" } })
  onChunk: ({ chunk }) => {
    // chunk is a UI Stream Protocol event ("part")
    process.stdout.write(JSON.stringify(norm(chunk)) + "\n");
  },
});

await result.finished; // ensure stream closes
