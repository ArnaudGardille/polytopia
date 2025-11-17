/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'polytopia-blue': '#3B82F6',
        'polytopia-red': '#EF4444',
        'polytopia-green': '#10B981',
        'polytopia-orange': '#F59E0B',
      },
    },
  },
  plugins: [],
}

