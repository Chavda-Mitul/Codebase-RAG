import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Code-RAG — GraphRAG + CRAG Pipeline',
  description: 'Adaptive GraphRAG + Corrective Agentic Pipeline for codebase analysis',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
