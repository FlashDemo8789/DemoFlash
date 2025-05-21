import type { Config } from "tailwindcss";
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: { extend: { colors: { primary: "#2563eb" } } },
  plugins: []
} satisfies Config; 