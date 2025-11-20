import React, { useState, useEffect } from 'react';
import { GraduationCap, Search, AlertCircle, CheckCircle, Loader2, Download, TrendingUp } from 'lucide-react';
import { SilkBackground } from '../components/SilkBackground';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import '../index.css';

const API_BASE_URL = 'http://localhost:5000/api';

interface Exam {
  id: number;
  name: string;
  url: string;
}

interface ExamType {
  id: string;
  name: string;
}

interface Branch {
  name: string;
  url: string;
}

interface Subject {
  code: string;
  name: string;
  internal: number;
  external: number;
  total: number;
  result: string;
  date: string;
  color?: string;
  grade?: string;
  grade_point?: number;
}

interface Results {
  student_info: {
    usn?: string;
    name?: string;
  };
  semester?: number;
  subjects: Subject[];
  analysis?: {
    sgpa: number;
    total_marks: number;
    total_subjects: number;
    passed_subjects: number;
    failed_subjects: number;
    failed_subject_names: { name: string; code: string }[];
    performance_message: string;
  };
}

function Fetcher() {
  const [usn, setUsn] = useState('');
  const [exams, setExams] = useState<Exam[]>([]);
  const [selectedExam, setSelectedExam] = useState<Exam | null>(null);
  
  // Multi-level selection state
  const [examTypes, setExamTypes] = useState<ExamType[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');
  const [finalExams, setFinalExams] = useState<Exam[]>([]);
  const [selectedFinalExam, setSelectedFinalExam] = useState<Exam | null>(null);
  const [selectionLevel, setSelectionLevel] = useState<'main' | 'type' | 'final' | 'ready'>('main');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<Results | null>(null);
  const [loadingExams, setLoadingExams] = useState(false);
  const [examsLoaded, setExamsLoaded] = useState(false);

  // Load exams only once on initial mount
  useEffect(() => {
    if (!examsLoaded && exams.length === 0) {
      fetchExams();
    }
  }, []);

  const fetchExams = async () => {
    if (loadingExams) return;
    
    setLoadingExams(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/exams`);
      const data = await response.json();
      if (data.success) {
        setExams(data.exams);
        setExamsLoaded(true);
      } else {
        setError('Failed to load exams');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
    } finally {
      setLoadingExams(false);
    }
  };

  const validateUSN = (value: string): boolean => {
    const usnPattern = /^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}$/;
    return usnPattern.test(value);
  };

  const handleUSNChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setUsn(value);
    setError('');
  };

  const handleExamSelection = async (exam: Exam) => {
    setSelectedExam(exam);
    setError('');
    setLoading(true);

    // Reset previous selections
    setExamTypes([]);
    setSelectedType('');
    setFinalExams([]);
    setSelectedFinalExam(null);

    try {
      const response = await fetch(`${API_BASE_URL}/check-exam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exam_url: exam.url }),
      });

      const data = await response.json();

      if (data.success && data.requires_selection) {
        setExamTypes(data.types);
        setSelectionLevel('type');
      } else if (data.success && !data.requires_selection) {
        setSelectionLevel('ready');
      } else {
        setError(data.error || 'Failed to load exam details');
      }
    } catch (err) {
      setError('Failed to load exam details');
    } finally {
      setLoading(false);
    }
  };

  const handleFetchResults = async () => {
    if (!usn || !validateUSN(usn)) {
      setError('Please enter a valid USN');
      return;
    }

    if (!selectedExam) {
      setError('Please select an exam');
      return;
    }

    if (examTypes.length > 0 && !selectedType) {
      setError('Please select exam type (CBCS or CBCS-RV)');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await fetch(`${API_BASE_URL}/fetch-results`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          usn,
          exam_url: selectedExam.url,
          exam_type: selectedType,
          fetch_details: true,
        }),
      });

      const data = await response.json();

      if (data.success) {
        setResults(data.results);
      } else if (data.error === 'invalid_usn') {
        setError(data.message || 'University Seat Number is not available or Invalid..!');
      } else {
        setError(data.message || 'Failed to fetch results');
      }
    } catch (err) {
      setError('Failed to fetch results. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getMarkColor = (total: number): string => {
    if (total >= 90) return '#10b981';
    if (total >= 80) return '#3b82f6';
    if (total >= 70) return '#8b5cf6';
    if (total >= 60) return '#f59e0b';
    if (total >= 50) return '#f97316';
    if (total >= 40) return '#eab308';
    return '#ef4444';
  };

  return (
    <div className="min-h-screen relative">
      <SilkBackground />
      
      <div className="container mx-auto px-4 py-6 relative z-10">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <GraduationCap className="w-12 h-12 text-green-600" />
            <h1 className="text-4xl font-bold text-gray-800">VTU Results Fetcher</h1>
          </div>
          <p className="text-gray-600">
            Fetch your VTU examination results with detailed analysis
          </p>
        </div>

        {/* Main Card */}
        <Card className="max-w-4xl mx-auto backdrop-blur-sm bg-white/90 shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="w-6 h-6" />
              Enter Your Details
            </CardTitle>
            <CardDescription>
              Select your exam and enter your USN to fetch results
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Show exam list ONLY when no exam is selected */}
            {!selectedExam && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium">Select Exam</label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={fetchExams}
                    disabled={loadingExams}
                  >
                    {loadingExams ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        Loading...
                      </>
                    ) : examsLoaded ? (
                      'Reload Exams'
                    ) : (
                      'Load Exams'
                    )}
                  </Button>
                </div>
                {loadingExams ? (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading exams...
                  </div>
                ) : exams.length > 0 ? (
                  <div className="space-y-2">
                    {exams.map((exam) => (
                      <button
                        key={exam.id}
                        onClick={() => handleExamSelection(exam)}
                        className="w-full text-left px-4 py-3 rounded-lg border-2 border-gray-200 hover:border-green-300 bg-white transition-all"
                      >
                        <div className="font-medium text-sm">{exam.name}</div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p className="text-sm mb-4">Click "Load Exams" to fetch available exams</p>
                  </div>
                )}
              </div>
            )}

            {/* Show input form when exam is selected */}
            {selectedExam && (
              <>
                {/* Loading indicator */}
                {loading && examTypes.length === 0 && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading exam details...
                  </div>
                )}

                {/* Show form only after exam details loaded */}
                {!loading && (
                  <>
                    {/* Selected exam display */}
                    <div className="bg-green-50 border-2 border-green-600 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-xs text-gray-600 mb-1">Selected Exam:</p>
                          <p className="font-medium text-sm">{selectedExam.name}</p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setSelectedExam(null);
                            setExamTypes([]);
                            setSelectedType('');
                            setUsn('');
                            setError('');
                          }}
                        >
                          Change Exam
                        </Button>
                      </div>
                    </div>

                    {/* USN Input */}
                    <div>
                      <label className="block text-sm font-medium mb-2">USN (University Seat Number)</label>
                      <Input
                        placeholder="e.g., 4PA23CS102"
                        value={usn}
                        onChange={handleUSNChange}
                        maxLength={10}
                        className="text-lg"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Format: #@@##@@### (e.g., 4PA23CS102, 2MN25EC021)
                      </p>
                    </div>

                    {/* Type Selection - Show if exam has types */}
                    {examTypes.length > 0 && (
                      <div>
                        <label className="block text-sm font-medium mb-2">Type (Original/Revaluation)</label>
                        <select
                          value={selectedType}
                          onChange={(e) => setSelectedType(e.target.value)}
                          className="w-full px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-green-600 bg-white"
                        >
                          <option value="">-- Select Type --</option>
                          {examTypes.map((type) => (
                            <option key={type.id} value={type.id}>
                              {type.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </>
                )}
              </>
            )}

            {/* Error Message */}
            {error && (
              <div className="flex items-start gap-2 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {/* Fetch Button - Only show when ALL required fields are filled */}
            {selectedExam && usn && validateUSN(usn) && 
             (examTypes.length === 0 || selectedType) && (
              <Button
                onClick={handleFetchResults}
                disabled={loading}
                className="w-full h-12 text-lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Fetching Results...
                  </>
                ) : (
                  <>
                    <Search className="w-5 h-5 mr-2" />
                    Fetch Results
                  </>
                )}
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Results Display */}
        {results && (
          <div className="max-w-6xl mx-auto mt-8 space-y-6">
            {/* Analysis Card */}
            {results.analysis && (
              <Card className="backdrop-blur-sm bg-white/90 shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-6 h-6 text-green-600" />
                    Performance Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-3xl font-bold text-green-600">
                        {results.analysis.sgpa}
                      </div>
                      <div className="text-sm text-gray-600">SGPA</div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-3xl font-bold text-blue-600">
                        {results.analysis.total_marks}
                      </div>
                      <div className="text-sm text-gray-600">Total Marks</div>
                    </div>
                    <div className="text-center p-4 bg-emerald-50 rounded-lg">
                      <div className="text-3xl font-bold text-emerald-600">
                        {results.analysis.passed_subjects}
                      </div>
                      <div className="text-sm text-gray-600">Passed</div>
                    </div>
                    <div className="text-center p-4 bg-red-50 rounded-lg">
                      <div className="text-3xl font-bold text-red-600">
                        {results.analysis.failed_subjects}
                      </div>
                      <div className="text-sm text-gray-600">Failed</div>
                    </div>
                  </div>

                  <div
                    className={`flex items-start gap-2 p-4 rounded-lg ${
                      results.analysis.failed_subjects > 0
                        ? 'bg-amber-50 border border-amber-200'
                        : 'bg-green-50 border border-green-200'
                    }`}
                  >
                    {results.analysis.failed_subjects > 0 ? (
                      <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                    ) : (
                      <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                    )}
                    <p
                      className={`text-sm ${
                        results.analysis.failed_subjects > 0
                          ? 'text-amber-800'
                          : 'text-green-800'
                      }`}
                    >
                      {results.analysis.performance_message}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Subjects Table */}
            <Card className="backdrop-blur-sm bg-white/90 shadow-xl">
              <CardHeader>
                <CardTitle>Subject-wise Results</CardTitle>
                <CardDescription>
                  {results.semester && `Semester: ${results.semester}`}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-3 font-semibold">Code</th>
                        <th className="text-left p-3 font-semibold">Subject Name</th>
                        <th className="text-center p-3 font-semibold">Internal</th>
                        <th className="text-center p-3 font-semibold">External</th>
                        <th className="text-center p-3 font-semibold">Total</th>
                        <th className="text-center p-3 font-semibold">Grade</th>
                        <th className="text-center p-3 font-semibold">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.subjects.map((subject, idx) => (
                        <tr
                          key={idx}
                          className="border-b hover:bg-gray-50 transition-colors"
                        >
                          <td className="p-3 font-medium">{subject.code}</td>
                          <td className="p-3">{subject.name}</td>
                          <td className="text-center p-3">{subject.internal}</td>
                          <td className="text-center p-3">{subject.external}</td>
                          <td className="text-center p-3">
                            <span
                              className="px-3 py-1 rounded-full text-white font-semibold"
                              style={{ backgroundColor: getMarkColor(subject.total) }}
                            >
                              {subject.total}
                            </span>
                          </td>
                          <td className="text-center p-3 font-semibold">
                            {subject.grade || '-'}
                          </td>
                          <td className="text-center p-3">
                            <span
                              className={`px-3 py-1 rounded-full text-xs font-semibold ${
                                subject.result.toUpperCase() === 'P'
                                  ? 'bg-green-100 text-green-800'
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {subject.result.toUpperCase() === 'P' ? 'PASS' : 'FAIL'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center py-8 text-gray-600 text-sm relative z-10">
        <p>© 2025 VTU Results Fetcher. Made with ❤️ for VTU Students</p>
      </footer>
    </div>
  );
}

export default Fetcher;
