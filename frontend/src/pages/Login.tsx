import React from 'react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';

const Login: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Sign in to your account</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium">Email</label>
            <Input placeholder="you@example.com" />
          </div>

          <div>
            <label className="block text-sm font-medium">Password</label>
            <Input type="password" placeholder="Password" />
          </div>

          <Button className="w-full">Sign In</Button>
        </div>

        <p className="text-sm text-gray-500 mt-4">Don't have an account? <a href="/register" className="text-blue-600">Register</a></p>
      </div>
    </div>
  );
};

export default Login;
