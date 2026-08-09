import { existsSync } from 'node:fs';
import { defineConfig } from '@playwright/test';

const port = 4321;

/**
 * Some sandboxes ship Chromium at a fixed path whose build number will not
 * match whatever `@playwright/test` currently pins. Use it when it is there
 * rather than downloading a second copy; fall back to Playwright's own browser
 * everywhere else, which is what CI installs.
 */
const preinstalledChromium = '/opt/pw-browsers/chromium';
const executablePath = existsSync(preinstalledChromium) ? preinstalledChromium : undefined;

export default defineConfig({
  testDir: './tests',
  // `.test.ts` files are vitest's; Playwright takes only the `.spec.ts`.
  testMatch: '**/*.spec.ts',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: `http://localhost:${port}`,
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium', launchOptions: { executablePath } } },
  ],
  // Serves `dist/`, so the tests need a build first — `npm test` runs one.
  webServer: {
    command: `npx astro preview --port ${port}`,
    url: `http://localhost:${port}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
