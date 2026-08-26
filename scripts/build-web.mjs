import * as esbuild from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const serve = process.argv.includes('--serve');

const buildOptions = {
  entryPoints: [path.join(root, 'web/app.ts')],
  outfile: path.join(root, 'web/dist/app.js'),
  bundle: true,
  format: 'esm',
  target: 'es2022',
  sourcemap: true,
  logLevel: 'info',
};

if (serve) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  const { host, port } = await ctx.serve({
    servedir: path.join(root, 'web'),
    port: 8080,
  });
  console.log(`Serving http://${host === '0.0.0.0' ? 'localhost' : host}:${port}`);
} else {
  await esbuild.build(buildOptions);
}
