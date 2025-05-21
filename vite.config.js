import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path"; // Use node:path import
import tsconfigPaths from "vite-tsconfig-paths"; // Keep this plugin

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  server: { 
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/report': 'http://localhost:8000'
    }
  },
}); 