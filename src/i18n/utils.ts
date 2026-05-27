import type { CollectionEntry } from "astro:content";
import { LOCALES, DEFAULT_LOCALE, type Locale } from "./ui";

/** Extract the locale from a post collection entry id (e.g. "zh/why-blog" → "zh") */
export function getLangFromPostId(id: string): Locale {
  const first = id.split("/")[0] as Locale;
  return LOCALES.includes(first) ? first : DEFAULT_LOCALE;
}

/**
 * Extract the filename slug from a post id, stripping locale and
 * organizational directories (year/month).
 * e.g. "zh/2025/03/why-blog" → "why-blog"
 */
export function getBaseSlug(id: string): string {
  return id.split("/").at(-1) ?? id;
}

/** Filter a posts array to only those belonging to the given locale */
export function getPostsByLang(
  posts: CollectionEntry<"blog">[],
  lang: Locale
): CollectionEntry<"blog">[] {
  return posts.filter(post => getLangFromPostId(post.id) === lang);
}

/**
 * Find the translation of a post in the other language.
 * Matches by base slug (same filename, different lang directory).
 */
export function getTranslation(
  post: CollectionEntry<"blog">,
  allPosts: CollectionEntry<"blog">[]
): CollectionEntry<"blog"> | undefined {
  const baseSlug = getBaseSlug(post.id);
  const currentLang = getLangFromPostId(post.id);
  return allPosts.find(
    p => getBaseSlug(p.id) === baseSlug && getLangFromPostId(p.id) !== currentLang
  );
}

/** Validate that a given string is a supported locale */
export function isValidLocale(lang: string): lang is Locale {
  return LOCALES.includes(lang as Locale);
}

/** Get the other locale (en ↔ zh) */
export function getOtherLocale(lang: Locale): Locale {
  return lang === "zh" ? "en" : "zh";
}
