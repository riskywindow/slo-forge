import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    port: 4174,
    strictPort: true,
  },
  test: {
    environment: "happy-dom",
    coverage: {
      reporter: ["text", "json-summary"],
    },
  },
});
