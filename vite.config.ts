import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Changed path import back to require for this specific fix attempt
const path = require("path");

// NOTE: tsconfigPaths plugin removed as per the specific instruction's config example

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "flashcamp/frontend/src") // Use absolute path for front-end src
    }
  }
}); 