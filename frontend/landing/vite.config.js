import { resolve } from "node:path";
import { defineConfig } from "vite";

// Two pages: the landing page and the app. VITE_API_URL points the app at the API.
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        app: resolve(import.meta.dirname, "app.html"),
      },
    },
  },
});
