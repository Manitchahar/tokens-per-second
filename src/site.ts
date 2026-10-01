// Site-wide settings. Change `repo` to your GitHub "owner/name" after you push,
// so the "Open in Colab" buttons point at your copy of the notebooks.
export const SITE = {
  name: 'Tokens per Second',
  tagline: 'Inference engineering from first principles, zero to hero.',
  repo: 'Manitchahar/tokens-per-second',
  branch: 'main',
};

const base = import.meta.env.BASE_URL.endsWith('/') ? import.meta.env.BASE_URL : import.meta.env.BASE_URL + '/';

/** Prefix an internal path with the deploy base ("/" locally). */
export function url(path: string) {
  return base + path.replace(/^\//, '');
}

export function colabUrl(notebook: string) {
  return `https://colab.research.google.com/github/${SITE.repo}/blob/${SITE.branch}/public/notebooks/${notebook}`;
}
