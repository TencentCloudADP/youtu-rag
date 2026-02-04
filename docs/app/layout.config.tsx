import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { i18n } from '@/lib/i18n';
import Image from 'next/image';
import { getAssetPath } from '@/lib/utils';

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
      title: (
        <div className="logo-container">
          <Image
            src={getAssetPath("/assets/youtu-rag-logo.png")}
            alt="Youtu-RAG"
            fill
            className="logo-image"
          />
        </div>
      ),
      url: '/'
    }
  };
}
