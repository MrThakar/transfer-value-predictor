import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // In development, forward API calls to the FastAPI server.
    proxy: {
      "/predict": API,
      "/metadata": API,
      "/comparables": API,
      "/health": API,
    },
  },
});
