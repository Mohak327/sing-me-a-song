import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The page asks the FastAPI server (server/app.py) for data and audio.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8010' } },
})
