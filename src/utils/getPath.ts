import { BLOG_PATH } from "@/content.config";
import { LOCALES } from "@/i18n/ui";
import { slugifyStr } from "./slugify";

/**
 * Get full path of a blog post.
 *
 * When the post lives under a locale subdirectory (e.g. src/data/blog/zh/),
 * the locale prefix is automatically detected from the entry `id` and
 * included in the returned path as /{lang}/posts/{slug}.
 *
 * @param id - collection entry id (e.g. "zh/why-blog")
 * @param filePath - the blog post full file location
 * @param includeBase - when false, returns only the slug segment (used for params in getStaticPaths)
 */
export function getPath(
  id: string,
  filePath: string | undefined,
  includeBase = true
) {
  // Detect locale prefix from id (e.g. "zh/why-blog" → "zh")
  const idParts = id.split("/");
  const detectedLang =
    idParts.length > 1 && LOCALES.includes(idParts[0] as (typeof LOCALES)[number])
      ? idParts[0]
      : undefined;

  // Build intermediate path segments from filePath, stripping locale dirs
  // and numeric-only segments (year/month organizational folders)
  const isNumericSegment = (s: string) => /^\d+$/.test(s);
  const pathSegments = filePath
    ?.replace(BLOG_PATH, "")
    .split("/")
    .filter(path => path !== "")
    .filter(path => !path.startsWith("_"))
    .filter(path => !LOCALES.includes(path as (typeof LOCALES)[number]))
    .filter(path => !isNumericSegment(path))
    .slice(0, -1) // remove filename
    .map(segment => slugifyStr(segment));

  // Making sure `id` does not contain the directory (use only the last segment)
  const slug = idParts.slice(-1);

  const langPrefix = detectedLang ? `/${detectedLang}` : "";
  const basePath = includeBase ? `${langPrefix}/posts` : "";

  // If not inside a sub-dir (beyond the locale dir), simply return the file path
  if (!pathSegments || pathSegments.length < 1) {
    return [basePath, slug].join("/");
  }

  return [basePath, ...pathSegments, slug].join("/");
}
