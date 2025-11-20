import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout'
// App (original fetcher) left in repo as `App.tsx`; we now use `Fetcher.tsx` for the fetcher page.
import Landing from './pages/Landing'
import Fetcher from './pages/Fetcher'
import Login from './pages/Login'
import Register from './pages/Register'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="fetcher" element={<Fetcher />} />
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
          <Route path="contact" element={<div className='min-h-screen flex items-center justify-center'>Contact page (placeholder)</div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
