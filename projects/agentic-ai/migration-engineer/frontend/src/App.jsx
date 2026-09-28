import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import MigrationConsole from './components/Console/MigrationConsole';
import MigrationHistory from './components/History/MigrationHistory';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/console" element={<MigrationConsole />} />
        <Route path="/history" element={<MigrationHistory />} />
        <Route path="/" element={<Navigate to="/console" replace />} />
        <Route path="*" element={<Navigate to="/console" replace />} />
      </Routes>
    </Layout>
  );
}
