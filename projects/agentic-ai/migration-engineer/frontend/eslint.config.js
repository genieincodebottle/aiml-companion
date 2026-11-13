// ESLint flat config for the Autonomous Migration Engineer frontend (React 19 + Vite).
// Formatting is owned by Prettier, so this focuses on correctness. Noisy
// stylistic rules are downgraded to warnings so CI fails only on real problems.
import js from '@eslint/js';
import globals from 'globals';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  react.configs.flat.recommended,
  reactHooks.configs.flat.recommended,
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { ...globals.browser, ...globals.node },
    },
    settings: { react: { version: 'detect' } },
    rules: {
      // Vite's automatic JSX runtime: React need not be in scope.
      'react/react-in-jsx-scope': 'off',
      'react/jsx-uses-react': 'off',
      // This codebase does not use prop-types by design.
      'react/prop-types': 'off',
      // Surface, do not block, on unused symbols.
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      // New in react-hooks v7 (React Compiler rules): flags the plain fetch-on-mount
      // pattern. Advisory, so a warning rather than a CI failure.
      'react-hooks/set-state-in-effect': 'warn',
    },
  },
];
