/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // TCM Water-Ink Gold palette
        base: "#0f2018",           // deepest bg — ink-black forest
        surface: "#152b1e",        // sidebar / panel
        "surface-2": "#1c3829",    // elevated cards
        "surface-3": "#243f33",    // hover / dividers
        "bubble-user": "#163d50",  // user message — teal contrast
        "bubble-assistant": "#172d21", // assistant — deep forest
        accent: "#c9a84c",         // cinnabar gold — primary CTA
        "accent-light": "#dfc068", // lighter gold — hover state
        jade: "#7aab8a",           // 青瓷 celadon — secondary
        muted: "#6a8a72",          // dimmed text
        paper: "#e8e0d0",          // main text — warm parchment
        vermillion: "#b84020",     // danger / stop
      },
      fontFamily: {
        sans: ["Noto Sans SC", "system-ui", "sans-serif"],
        serif: ["Noto Serif SC", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
