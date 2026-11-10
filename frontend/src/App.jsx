import { useState } from 'react';
import { Menu } from './components/Menu';


export default function App() {
  const [selectedTaskType, setSelectedTaskType] = useState('v-kis');
  const [selectedDataset, setSelectedDataset] = useState('mvk');

  const onTaskTypeChange = (taskType) => {
    setSelectedTaskType(taskType);
  }

  const onDatasetChange = (dataset) => {
    setSelectedDataset(dataset);
  }

  return (
    <main className="app-shell" data-task-type={selectedTaskType} data-dataset={selectedDataset}>
      <Menu onTaskTypeChange={onTaskTypeChange} onDatasetChange={onDatasetChange} />
    </main>
  );
}
