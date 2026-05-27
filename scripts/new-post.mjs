#!/usr/bin/env node

/**
 * Interactive CLI to scaffold a new blog post.
 *
 * Usage:
 *   npm run new-post                          (interactive prompts)
 *   npm run new-post -- --title "My Post" --lang en --tags "tech,ai" --description "About AI"
 *
 * Creates a markdown file at:
 *   src/data/blog/{lang}/{YYYY}/{MM}/{slug}.md
 */

import { createInterface } from "node:readline";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

function slugify(str) {
  return str
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    if (argv[i].startsWith("--") && i + 1 < argv.length) {
      const key = argv[i].slice(2);
      args[key] = argv[++i];
    }
  }
  return args;
}

async function main() {
  const cliArgs = parseArgs(process.argv);
  const isInteractive = !cliArgs.title;

  let rl;
  const ask = isInteractive
    ? (q) => {
        if (!rl) rl = createInterface({ input: process.stdin, output: process.stdout });
        return new Promise(resolve => rl.question(q, resolve));
      }
    : () => Promise.resolve("");

  if (isInteractive) console.log("\n📝  New Blog Post\n");

  const title = cliArgs.title || (await ask("Title: "));
  if (!title.trim()) {
    console.error("Title is required. Pass --title or run interactively.");
    process.exit(1);
  }

  const defaultSlug = slugify(title);
  const slug = cliArgs.slug || (isInteractive ? (await ask(`Slug [${defaultSlug}]: `)).trim() || defaultSlug : defaultSlug);

  const lang = cliArgs.lang && ["zh", "en"].includes(cliArgs.lang)
    ? cliArgs.lang
    : isInteractive
      ? (l => ["zh", "en"].includes(l) ? l : "zh")((await ask("Language (zh/en) [zh]: ")).trim())
      : "zh";

  const tagsRaw = cliArgs.tags || (isInteractive ? await ask("Tags (comma-separated) [others]: ") : "");
  const tags = tagsRaw.trim()
    ? tagsRaw.split(",").map(t => t.trim()).filter(Boolean)
    : ["others"];

  const description = cliArgs.description || (isInteractive ? await ask("Description: ") : title);

  const draft = cliArgs.draft === "true" || (isInteractive && (await ask("Draft? (y/n) [n]: ")).trim().toLowerCase() === "y");

  const now = new Date();
  const year = String(now.getFullYear());
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const isoDate = now.toISOString();

  const dir = join("src", "data", "blog", lang, year, month);
  const filePath = join(dir, `${slug}.md`);

  if (existsSync(filePath)) {
    console.error(`\n❌  File already exists: ${filePath}`);
    process.exit(1);
  }

  const tagsYaml = tags.map(t => `  - ${t}`).join("\n");

  const frontmatter = `---
author: Yuhao Chen
pubDatetime: ${isoDate}
title: "${title.replace(/"/g, '\\"')}"
slug: ${slug}
featured: false
draft: ${draft}
tags:
${tagsYaml}
description: "${(description || title).replace(/"/g, '\\"')}"
---

`;

  mkdirSync(dir, { recursive: true });
  writeFileSync(filePath, frontmatter, "utf-8");

  console.log(`\n✅  Created: ${filePath}`);
  console.log(`    URL:     /${lang}/posts/${slug}/\n`);

  if (rl) rl.close();
}

main();
