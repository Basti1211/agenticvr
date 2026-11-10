import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import type { SourceMapInput } from 'rollup';

const root = process.cwd();

type CssSourceMap = SourceMapInput & {
  sources: string[];
};

function normalizeCssSourceMaps(): Plugin {
  return {
    name: 'normalize-css-sourcemaps',
    enforce: 'post',
    apply: 'serve', // oder 'build' oder weglassen
    async transform(code, id) {
      if (!/\.css($|\?)/.test(id)) return null;

      const context = this as {
        getCombinedSourcemap?: () => CssSourceMap | null;
      };
      const map = context.getCombinedSourcemap?.();
      if (!map || !Array.isArray(map.sources)) return null;

      map.sources = map.sources.map((s: string) => {
        const normalized = s.replace(/\\/g, '/');
        const rootNorm = root.replace(/\\/g, '/');

        if (normalized.startsWith(rootNorm)) {
          return path.relative(root, normalized).replace(/\\/g, '/');
        }
        return s;
      });

      return { code, map: map ?? null };
    },
  };
}

function securityHeadersPlugin(): Plugin {
  return {
    name: 'security-headers',
    configResolved() {
      // Plugin setup
    },
    transformIndexHtml: {
      order: 'pre',
      handler(html) {
        return html.replace(
          '</head>',
          `
    <!-- Security Headers Added by Vite -->
    <meta name="referrer" content="strict-origin-when-cross-origin">
    </head>`
        );
      },
    },
  };
}

export default defineConfig({
  resolve: {
    alias: {
      '@': __dirname,
    },
  },
  build: {
    assetsInlineLimit: 0,
    cssCodeSplit: true,
    sourcemap: true, // Only in dev to reduce prod bundle - TODO: change it for prod
    // Security: Generate hash-based output names to prevent caching issues
    rollupOptions: {
      output: {
        entryFileNames: '[name].[hash].js',
        chunkFileNames: '[name].[hash].js',
        assetFileNames: '[name].[hash][extname]',
      },
    },
  },
  css: {
    devSourcemap: true,
  },
  plugins: [
    react(),
    normalizeCssSourceMaps(),
    securityHeadersPlugin(),
  ],
  server: {
    proxy: {
      "/api": {
        target: process.env.BACKEND_URL || "http://localhost:5001",
        ws: false,
        changeOrigin: true,
      }
    },
  }
})
