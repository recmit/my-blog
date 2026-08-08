/**
 * `/feed.xml` — the same path Jekyll's `jekyll-feed` published, so existing
 * subscribers keep working. The format changes from Atom to RSS 2.0, which is
 * what `@astrojs/rss` emits; readers handle both.
 */
import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';
import { getPublishedPosts, postPath } from '../lib/posts';

export const GET: APIRoute = async (context) => {
  const posts = await getPublishedPosts();

  return rss({
    title: "David Recio's Blog",
    description: 'Posts on statistics, machine learning and data analysis.',
    // `context.site` comes from `site` in astro.config.mjs, which is set.
    site: context.site!,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: postPath(post),
    })),
  });
};
