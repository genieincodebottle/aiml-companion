// ESLint flat config for the Mission Control console (React 19 + Vite).
// Formatting is owned by Prettier, so this focuses on correctness. Noisy
// stylistic rules are downgraded to warnings so CI fails only on real problems.
import js from '@eslint/js';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

export default [
  { ignores: ['dist/', 'node_modules/'] },
  js.configs.recommended,
  react.configs.flat.recommended,
  react.configs.flat['jsx-runtime'], // Vite's automatic JSX runtime: React need not be in scope
  reactHooks.configs.flat.recommended,
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
    },
    settings: { react: { version: 'detect' } },
    rules: {
      // This codebase does not use prop-types by design.
      'react/prop-types': 'off',
      // Surface, do not block, on unused symbols.
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },
];
