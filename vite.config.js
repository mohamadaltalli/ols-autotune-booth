import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from https://<user>.github.io/<repo>/, so
// every asset URL needs that prefix. Without it you get a blank page and 404s.
//
// PAGES_BASE is passed in explicitly by .github/workflows/deploy.yml.
// GITHUB_REPOSITORY is the fallback: GitHub Actions sets it on every runner
// automatically, so builds still resolve correctly if the env line is dropped.
// Locally neither is set, so the base stays '/'.
const slug = process.env.PAGES_BASE || process.env.GITHUB_REPOSITORY || ''
const repo = slug.split('/')[1] || ''

// A repo literally named <user>.github.io is served from the domain root, and
// so is a custom domain. Both want '/'.
const base = repo && !repo.endsWith('.github.io') ? `/${repo}/` : '/'

console.log(`[vite] building with base "${base}"`)

export default defineConfig({
  base,
  plugins: [react()],
})
