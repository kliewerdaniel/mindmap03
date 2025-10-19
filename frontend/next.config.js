/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // Disable ESLint during build to allow for containerization
    ignoreDuringBuilds: true,
  },
  output: 'standalone',
  experimental: {
    turbo: {
      resolveAlias: {
        '@/*': './*',
      },
    },
  },
};

module.exports = nextConfig;
