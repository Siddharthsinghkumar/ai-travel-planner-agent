/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        glassAccent: "rgba(56,189,248,0.85)",
        "slate-void": "#0f172a",
        "cyan-glow": "rgba(56,189,248,0.15)",
        "aurora-blue": "#bae6fd",
        "aurora-blush": "#fbcfe8"
      },
      boxShadow: {
        "glass-dark": "0 8px 32px rgba(0,0,0,0.4)",
        "glass-light": "0 20px 40px -10px rgba(0,0,0,0.05)",
        "neo-inset": "inset 2px 2px 5px rgba(0,0,0,0.2), inset -2px -2px 5px rgba(255,255,255,0.1)"
      }
    }
  },
  plugins: []
}
