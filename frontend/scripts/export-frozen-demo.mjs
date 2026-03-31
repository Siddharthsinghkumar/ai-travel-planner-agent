#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(__dirname, "..");
const distDir = resolve(projectDir, "dist");
const indexPath = resolve(distDir, "index.html");
const outputPath = resolve(distDir, "frozen-demo.html");

if (!existsSync(indexPath)) {
  console.error("dist/index.html not found. Run `npm run build` first.");
  process.exit(1);
}

let html = readFileSync(indexPath, "utf8");

const inlineCssBlocks = [];
const inlineJsBlocks = [];

html = html.replace(/<link\s+[^>]*rel=["']modulepreload["'][^>]*>\s*/gi, "");

html = html.replace(
  /<link\s+[^>]*rel=["']stylesheet["'][^>]*href=["']([^"']+)["'][^>]*>\s*/gi,
  (_full, href) => {
    if (/^(https?:)?\/\//i.test(href)) return "";
    const clean = String(href).replace(/^\//, "");
    const filePath = resolve(distDir, clean);
    const css = readFileSync(filePath, "utf8");
    inlineCssBlocks.push(`/* ${clean} */\n${css}`);
    return "";
  },
);

html = html.replace(
  /<script\s+[^>]*type=["']module["'][^>]*src=["']([^"']+)["'][^>]*><\/script>\s*/gi,
  (_full, src) => {
    if (/^(https?:)?\/\//i.test(src)) return "";
    const clean = String(src).replace(/^\//, "");
    const filePath = resolve(distDir, clean);
    const js = readFileSync(filePath, "utf8");
    inlineJsBlocks.push(`/* ${clean} */\n${js}`);
    return "";
  },
);

if (!inlineCssBlocks.length || !inlineJsBlocks.length) {
  console.error("Could not inline build assets from dist/index.html.");
  process.exit(1);
}

const inlineStylesTag = `<style id="frozen-inline-css">\n${inlineCssBlocks.join("\n\n")}\n</style>`;
const inlineScriptTag = `<script type="module" id="frozen-inline-js">\n${inlineJsBlocks
  .join("\n\n")
  .replace(/<\/script>/gi, "<\\/script>")}\n</script>`;

if (html.includes("</head>")) {
  html = html.replace("</head>", `${inlineStylesTag}\n</head>`);
} else {
  html = `${inlineStylesTag}\n${html}`;
}

if (html.includes("</body>")) {
  html = html.replace("</body>", `${inlineScriptTag}\n</body>`);
} else {
  html = `${html}\n${inlineScriptTag}\n`;
}

writeFileSync(outputPath, html, "utf8");
console.log(`Wrote ${outputPath}`);
