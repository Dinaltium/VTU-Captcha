import React from 'react';
import { Link } from 'react-router-dom';

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white">
      <div className="text-center py-16">
        <h1 className="text-6xl font-extralight tracking-tight mb-6">VTU Results Fetcher</h1>
        <p className="text-2xl text-gray-600 mb-8">Fetch Your Results Easily</p>

        <div className="flex items-center justify-center gap-4">
          <Link to="/login" className="inline-block bg-black text-white px-6 py-2 rounded-full font-semibold shadow-md">LOGIN</Link>
          <Link to="/register" className="inline-block bg-black text-white px-6 py-2 rounded-full font-semibold shadow-md">REGISTER</Link>
        </div>
      </div>
    </div>
  );
};

export default Landing;
