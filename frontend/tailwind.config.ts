import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // All semantic tokens map to CSS variables defined in globals.css.
        // The values use space-separated channels so Tailwind's <alpha-value>
        // syntax works (e.g. text-ink/70).
        canvas:        "rgb(var(--c-canvas) / <alpha-value>)",
        ink:           "rgb(var(--c-ink) / <alpha-value>)",
        text:          "rgb(var(--c-text) / <alpha-value>)",
        muted:         "rgb(var(--c-muted) / <alpha-value>)",
        subtle:        "rgb(var(--c-subtle) / <alpha-value>)",
        border:        "rgb(var(--c-border) / <alpha-value>)",
        line:          "rgb(var(--c-line) / <alpha-value>)",
        accent:        "rgb(var(--c-accent) / <alpha-value>)",
        "accent-soft": "rgb(var(--c-accent-soft) / <alpha-value>)",
        flood:         "rgb(var(--c-flood) / <alpha-value>)",
        permanent:     "rgb(var(--c-permanent) / <alpha-value>)",
        success:       "rgb(var(--c-success) / <alpha-value>)",
        warn:          "rgb(var(--c-warn) / <alpha-value>)",
        danger:        "rgb(var(--c-danger) / <alpha-value>)",
      },
      fontFamily: {
        sans:  ["var(--font-sans)",  "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "ui-serif", "Georgia", "serif"],
        mono:  ["var(--font-mono)",  "ui-monospace", "Menlo", "monospace"],
      },
      letterSpacing: {
        label: "0.08em",
        kicker: "0.14em",
      },
      borderRadius: {
        DEFAULT: "0.5rem",
        sm: "0.375rem",
        lg: "0.75rem",
        xl: "1rem",
        "2xl": "1.25rem",
      },
      boxShadow: {
        // Layered ambient shadows (no harsh drop-shadows)
        soft:  "0 1px 2px rgb(15 23 42 / 0.04), 0 1px 1px rgb(15 23 42 / 0.06)",
        glass: "0 8px 32px rgb(15 23 42 / 0.08), 0 2px 8px rgb(15 23 42 / 0.04), inset 0 1px 0 rgb(255 255 255 / 0.4)",
        "glass-dark": "0 8px 32px rgb(0 0 0 / 0.5), 0 2px 8px rgb(0 0 0 / 0.3), inset 0 1px 0 rgb(255 255 255 / 0.06)",
        focus: "0 0 0 3px rgb(var(--c-accent) / 0.25)",
        cta:   "0 6px 20px rgb(var(--c-accent) / 0.35), inset 0 1px 0 rgb(255 255 255 / 0.18)",
      },
      backgroundImage: {
        "accent-grad": "linear-gradient(135deg, rgb(var(--c-accent)) 0%, rgb(var(--c-accent-grad-end)) 100%)",
        "panel-sheen": "linear-gradient(180deg, rgb(255 255 255 / 0.08) 0%, transparent 60%)",
      },
      keyframes: {
        "fade-in":      { from: { opacity: "0" },                       to: { opacity: "1" } },
        "slide-in-l":   { from: { opacity: "0", transform: "translateX(-12px)" }, to: { opacity: "1", transform: "translateX(0)" } },
        "slide-in-r":   { from: { opacity: "0", transform: "translateX(12px)" },  to: { opacity: "1", transform: "translateX(0)" } },
        "slide-in-t":   { from: { opacity: "0", transform: "translateY(-8px)" },  to: { opacity: "1", transform: "translateY(0)" } },
        "pulse-dot":    {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%":      { opacity: "0.55", transform: "scale(0.92)" },
        },
        "shimmer":      {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "spin-slow":    { to: { transform: "rotate(360deg)" } },
      },
      animation: {
        "fade-in":     "fade-in 220ms cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-in-l":  "slide-in-l 280ms cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-in-r":  "slide-in-r 280ms cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-in-t":  "slide-in-t 220ms cubic-bezier(0.16, 1, 0.3, 1)",
        "pulse-dot":   "pulse-dot 1.4s ease-in-out infinite",
        "shimmer":     "shimmer 2.4s linear infinite",
        "spin-slow":   "spin-slow 6s linear infinite",
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
