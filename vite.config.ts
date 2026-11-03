import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173 },
  build: {
    target: "es2022",
    rolldownOptions: { input: { main: "index.html", robot: "robot.html" } },
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
} as Parameters<typeof defineConfig>[0]);
