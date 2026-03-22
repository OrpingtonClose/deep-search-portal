import './App.css'
import { Header } from './components/layout/Header';
import { Dashboard } from './components/layout/Dashboard';

function App() {
  return (
    <div className="h-screen w-screen flex flex-col bg-gray-950 text-gray-100 overflow-hidden">
      <Header />
      <Dashboard />
    </div>
  );
}

export default App
