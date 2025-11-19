import React from 'react';

export const SilkBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      <svg
        className="absolute inset-0 h-full w-full"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id="silk-pattern"
            x="0"
            y="0"
            width="100"
            height="100"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M0 50 Q 25 25, 50 50 T 100 50"
              stroke="rgba(34, 197, 94, 0.1)"
              strokeWidth="2"
              fill="none"
            />
            <path
              d="M0 50 Q 25 75, 50 50 T 100 50"
              stroke="rgba(34, 197, 94, 0.1)"
              strokeWidth="2"
              fill="none"
            />
          </pattern>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(16, 185, 129, 0.05)" />
            <stop offset="50%" stopColor="rgba(34, 197, 94, 0.03)" />
            <stop offset="100%" stopColor="rgba(5, 150, 105, 0.05)" />
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#gradient)" />
        <rect width="100%" height="100%" fill="url(#silk-pattern)" />
      </svg>
      <div className="absolute inset-0 bg-gradient-to-br from-green-50/50 via-white to-emerald-50/50" />
    </div>
  );
};
