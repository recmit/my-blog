import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

/**
 * The published path of a post: `/YYYY/MM/DD/<slug>.html`.
 *
 * These URLs are inherited from Jekyll and are load-bearing — inbound links
 * depend on them, and `tests/urls.test.ts` enforces the exact list. Dates are
 * read in UTC so a build machine west of Greenwich cannot shift a post's
 * directory back by a day.
 */
export function postPath(post: Post): string {
  return `/${postPathname(post)}.html`;
}

/**
 * The `[...slug]` param for a post: the published path without its leading
 * slash or its `.html`. The extension is added back by `build.format:
 * 'preserve'`, which writes `<param>.html` for a non-index route.
 */
export function postPathname(post: Post): string {
  const [year, month, day] = post.data.date.toISOString().slice(0, 10).split('-');
  return `${year}/${month}/${day}/${post.id}`;
}

/** Posts that should appear in listings, newest first. */
export async function getPublishedPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

/**
 * Estimated reading time in whole minutes, at 200 wpm.
 *
 * Counted over prose only: fenced code blocks and the `<details>` wrappers
 * around them are stripped first. These are notebook posts, so counting the
 * raw body would triple the estimate on the strength of dataframe dumps and
 * import lists that nobody reads top to bottom.
 */
export function readingTime(post: Post): number {
  const prose = post.body
    ?.replace(/^```[\s\S]*?^```/gm, '')
    .replace(/<[^>]+>/g, '')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '');
  const words = prose?.split(/\s+/).filter(Boolean).length ?? 0;
  return Math.max(1, Math.round(words / 200));
}

/** `Mar 19, 2022` — the format the Jekyll site used, pinned to UTC. */
export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
}
