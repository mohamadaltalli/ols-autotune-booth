import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from https://<user>.github.io/<repo>/,
// so every asset URL needs that prefix or you get a blank page and 404s in the
// console. This reads the repo name from the environment GitHub Actions already
// provides, so there is nothing to edit by hand.
//
// User/org sites (a repo literally named <user>.github.io) are served from the
// domain root, and so is a custom domain — both want '/'.
const repo = (process.env.GITHUB_REPOSITORY || '').split('/')[1] || ''
const base = repo && !repo.endsWith('.github.io') ? `/${repo}/` : '/'

export default defineConfig({
  base,
  plugins: [react()],
})
