#!/usr/bin/env node
import { execSync } from "child_process";
import { readFileSync, writeFileSync } from "fs";

export function buildPage({
  name,
  srcDir,
  manifestPath,
  templatePath,
  lang,
  title,
}) {
  console.log(`Installing dependencies...`);
  execSync("npm install", { cwd: srcDir, stdio: "inherit" });

  console.log(`Building ${name} page...`);
  execSync("npx vite build", { cwd: srcDir, stdio: "inherit" });

  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));

  const jsEntry = Object.values(manifest).find((e) => e.isEntry);
  if (!jsEntry) {
    console.error("manifest에서 entry 파일을 찾을 수 없습니다.");
    process.exit(1);
  }

  const jsPath = `/static/${name}/${jsEntry.file}`;
  const cssPath = jsEntry.css?.[0] ? `/static/${name}/${jsEntry.css[0]}` : null;

  if (!cssPath) {
    console.error("manifest에서 CSS 파일을 찾을 수 없습니다.");
    process.exit(1);
  }

  const html = `<!doctype html>
<html lang="${lang}">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <script
      type="module"
      crossorigin
      src="${jsPath}"
    ></script>
    <link rel="stylesheet" crossorigin href="${cssPath}" />
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
`;

  writeFileSync(templatePath, html, "utf-8");
  console.log(`${templatePath} updated`);
  console.log(`  JS:  ${jsPath}`);
  console.log(`  CSS: ${cssPath}`);
}
