/** @type {import('tailwindcss').Config} */
module.exports = {
  purge: [
    "./_includes/**/*.html",
    "./_layouts/**/*.html",
    "./_posts/*.html",
    "./*.html",
  ],
  darkMode: "media", // or 'media' or 'class'
  theme: {
    extend: {},
    fontFamily: {
      sans: ['"Inter"', "serif"],
      body: ["Inter", "serif"],
      display: ["Inter", "serif"],
    },
  },
  variants: {
    extend: {},
  },
  plugins: [
    require("@tailwindcss/typography"),
    require("daisyui"),
    function ({ addVariant }) {
      addVariant("child", "& > *");
      addVariant("child-hover", "& > *:hover");
    },
  ],
  daisyui: {
    styled: true,
    themes: [
      {
        light: {
          primary: "#0D2C69",
          secondary: "#475F8D",
          accent: "#48DB72",
          neutral: "#ECECEC",
          "base-100": "#FFFFFF",
          info: "#FFFFFF",
          success: "#48DB72",
          warning: "#FBBD23",
          error: "#F87272",
        },
      },
    ],
  },
};
