import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { i18n } from '@/lib/i18n';
import { LogoLink } from '@/components/logo-link';

/**
 * Shared layout configurations
 *
 * you can customise layouts individually from:
 * Home Layout: app/[lang]/(home)/layout.tsx
 * Docs Layout: app/[lang]/docs/layout.tsx
 */
export function baseOptions(locale: string): BaseLayoutProps {
  return {
    i18n,
    nav: {
      title: <LogoLink />,
      url: '/about.html'
    }
  };
}
