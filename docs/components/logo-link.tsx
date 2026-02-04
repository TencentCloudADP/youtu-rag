'use client';

import Image from 'next/image';
import { getAssetPath } from '@/lib/utils';

export function LogoLink() {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    window.location.href = '/about.html';
  };

  return (
    <div 
      className="logo-container"
      onClick={handleClick}
      style={{ cursor: 'pointer' }}
    >
      <Image
        src={getAssetPath("/assets/youtu-rag-logo.png")}
        alt="Youtu-RAG"
        fill
        className="logo-image"
      />
    </div>
  );
}
