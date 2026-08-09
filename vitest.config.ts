import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Only the `.test.ts` files. `tests/smoke.spec.ts` is Playwright's — it is
    // the one test here that needs a browser and a running server, and vitest
    // would otherwise claim it by its default `{test,spec}` pattern and fail
    // on the `@playwright/test` import.
    include: ['tests/**/*.test.ts'],
  },
});
