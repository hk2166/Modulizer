import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  server: {
    host: "127.0.0.1",
    port: 1420,
    strictPort: true,
    watch: {
      // Vite recursively watches every file under the project root. Our repo
      // contains a Python venv with thousands of torch header files, which
      // blows past the OS inotify watch limit (ENOSPC). We only need to watch
      // the actual frontend source, so ignore everything heavy/irrelevant.
      ignored: [
        "**/venv/**",
        "**/node_modules/**",
        "**/src-tauri/**",
        "**/dist/**",
        "**/backend/**",
        "**/data/**",
        "**/models/**",
        "**/.git/**",
        "**/__pycache__/**",
      ],
    },
  },
});
