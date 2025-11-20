import React from 'react';
import { Outlet } from 'react-router-dom';
import PillNav from './PillNav';
const logo = new URL('../logo.svg', import.meta.url).href;

const Layout: React.FC = () => {

  return (
    <div>
      <PillNav
      logo={logo}
  items={[
    { label: 'Home', href: '/' },
    { label: 'Fetcher', href: '/fetcher' },
    { label: 'Contact', href: '/contact' },
    { label: 'Login', href: '/login' },
    { label: 'Register', href: '/register' }
  ]}
  className="custom-nav"
  ease="power2.easeOut"
  baseColor="#000000"
  pillColor="#ffffff"
  hoveredPillTextColor="#ffffff"
  pillTextColor="#000000"
/>

      <main className="pt-16">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
