import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// NOTE: tsconfigPaths plugin removed as per the specific instruction's config example

export default defineConfig({
  plugins: [react()],
  server: { 
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/report': 'http://localhost:8000'
    }
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src") // Now __dirname is flashcamp/frontend
    }
  }
}); 