import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': 'http://localhost:3001',
      '/recordings': 'http://localhost:3001'
    }
  }
});
