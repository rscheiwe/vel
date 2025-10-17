// run: node scripts/basic_chat/ai_sdk_trace.mjs --prompt "Hello" > traces/ai.jsonl
import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get script directory and load .env from repository root
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..', '..');
dotenv.config({ path: join(rootDir, '.env') });

import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";

const argv = yargs(hideBin(process.argv)).option("prompt",{type:"string",demandOption:true}).argv;

const model = openai.chat("gpt-4o");

// Convert fullStream parts to wire protocol format
// AI SDK bug: fullStream parts use 'text'/'textDelta' internally, but wire protocol uses 'delta'
const norm = (part) => {
  const wireEvent = { type: part.type };

  // Copy id if present
  if (part.id !== undefined) wireEvent.id = part.id;

  // Convert text/textDelta to delta (wire protocol uses 'delta')
  if (part.type === 'text-delta' || part.type === 'reasoning-delta') {
    wireEvent.delta = part.delta || part.text || part.textDelta || '';
  }

  // Copy other fields as-is
  for (const [key, value] of Object.entries(part)) {
    if (key !== 'type' && key !== 'id' && key !== 'text' && key !== 'textDelta' && key !== 'delta') {
      wireEvent[key] = value;
    }
  }

  return wireEvent;
};

const events = [];

try {
  const result = await streamText({
    model,
    prompt: argv.prompt,
  });

  // Consume fullStream to collect all protocol events
  for await (const part of result.fullStream) {
    const normalized = norm(part);
    events.push(normalized);
  }

  // Write all events to stdout
  for (const event of events) {
    console.log(JSON.stringify(event));
  }
} catch (error) {
  console.error("Error:", error.message);
  process.exit(1);
}
