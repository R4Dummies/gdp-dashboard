import React, { useState, useRef } from 'react';
import { AlertCircle, Upload, Loader2, CheckCircle2, Edit3, Save, X, FileUp, ChevronDown, BarChart, TrendingUp, Target, Activity, Copy } from 'lucide-react';

export default function DictionaryClassificationBot() {
  const [csvData, setCsvData] = useState([]);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [textColumn, setTextColumn] = useState('');
  const [groundTruthColumn, setGroundTruthColumn] = useState('');
  const [dictionary, setDictionary] = useState([]);
  const [isEditingDictionary, setIsEditingDictionary] = useState(true);
  const [dictionaryText, setDictionaryText] = useState('');
  const [classificationResults, setClassificationResults] = useState([]);
  const [overallMetrics, setOverallMetrics] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [showAllFalsePositives, setShowAllFalsePositives] = useState(false);
  const [showAllFalseNegatives, setShowAllFalseNegatives] = useState(false);
  const [keywordAnalysis, setKeywordAnalysis] = useState({
    byRecall: [],
    byPrecision: [],
    byF1: []
  });
  const [activeMetric, setActiveMetric] = useState('recall');
  const fileInputRef = useRef(null);

  // Sample data for demonstration
  const sampleCsv = `ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want (, , , , etc) & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some "wardrobe crunches," and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.,0`;

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        parseCsv(event.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handleCsvInput = (e) => {
    const text = e.target.value;
    parseCsv(text);
  };

  const parseCsv = (text) => {
    try {
      const lines = text.trim().split('\n');
      if (lines.length < 2) {
        setError('CSV must have at least a header and one data row');
        return;
      }
      
      const headers = lines[0].split(',').map(h => h.trim());
      setCsvHeaders(headers);
      
      // Auto-detect columns if possible
      if (headers.includes('Statement') && !textColumn) {
        setTextColumn('Statement');
      }
      if (headers.includes('Answer') && !groundTruthColumn) {
        setGroundTruthColumn('Answer');
      }
      
      const data = [];
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index] ? values[index].trim() : '';
        });
        data.push(row);
      }
      
      setCsvData(data);
      setError('');
    } catch (err) {
      setError('Error parsing CSV: ' + err.message);
    }
  };

  const saveDictionary = () => {
    // Parse the format: "word1","word2","word3"
    const keywords = dictionaryText
      .match(/"[^"]+"/g)
      ?.map(k => k.replace(/"/g, '').trim())
      .filter(k => k.length > 0) || [];
    
    if (keywords.length === 0) {
      // Try parsing as comma-separated without quotes
      const altKeywords = dictionaryText.split(',')
        .map(k => k.trim())
        .filter(k => k.length > 0);
      if (altKeywords.length > 0) {
        setDictionary(altKeywords);
      }
    } else {
      setDictionary(keywords);
    }
    setIsEditingDictionary(false);
  };

  const classifyStatements = async () => {
    if (dictionary.length === 0) {
      setError('Please add keywords to the dictionary first');
      return;
    }
    
    if (csvData.length === 0) {
      setError('Please input CSV data first');
      return;
    }
    
    if (!textColumn) {
      setError('Please select the text column for analysis');
      return;
    }
    
    setIsClassifying(true);
    setError('');
    
    try {
      let truePositives = 0;
      let falsePositives = 0;
      let falseNegatives = 0;
      let trueNegatives = 0;
      
      const results = csvData.map(row => {
        const statement = (row[textColumn] || '').toLowerCase();
        const matchedKeywords = dictionary.filter(keyword => 
          statement.includes(keyword.toLowerCase())
        );
        
        const predicted = matchedKeywords.length > 0 ? 1 : 0;
        const groundTruth = groundTruthColumn && row[groundTruthColumn] !== undefined ? 
          parseInt(row[groundTruthColumn]) : null;
        
        let category = null;
        if (groundTruth !== null) {
          if (predicted === 1 && groundTruth === 1) {
            truePositives++;
            category = 'TP';
          } else if (predicted === 1 && groundTruth === 0) {
            falsePositives++;
            category = 'FP';
          } else if (predicted === 0 && groundTruth === 1) {
            falseNegatives++;
            category = 'FN';
          } else {
            trueNegatives++;
            category = 'TN';
          }
        }
        
        return {
          ...row,
          predicted,
          groundTruth,
          category,
          matchedKeywords,
          score: matchedKeywords.length
        };
      });
      
      setClassificationResults(results);
      
      // Calculate overall metrics
      if (groundTruthColumn) {
        const precision = (truePositives + falsePositives) > 0 ? 
          truePositives / (truePositives + falsePositives) : 0;
        const recall = (truePositives + falseNegatives) > 0 ? 
          truePositives / (truePositives + falseNegatives) : 0;
        const f1 = precision > 0 && recall > 0 ? 
          2 * precision * recall / (precision + recall) : 0;
        const accuracy = (truePositives + trueNegatives) / 
          (truePositives + trueNegatives + falsePositives + falseNegatives);
        
        setOverallMetrics({
          precision: (precision * 100).toFixed(2),
          recall: (recall * 100).toFixed(2),
          f1Score: (f1 * 100).toFixed(2),
          accuracy: (accuracy * 100).toFixed(2),
          truePositives,
          falsePositives,
          falseNegatives,
          trueNegatives
        });
      }
    } catch (err) {
      setError('Error classifying statements: ' + err.message);
    } finally {
      setIsClassifying(false);
    }
  };

  const analyzeKeywords = async () => {
    if (!groundTruthColumn || !textColumn) {
      setError('Ground truth column is required for keyword analysis');
      return;
    }
    
    setIsAnalyzing(true);
    setError('');
    
    try {
      // Calculate metrics for each keyword
      const keywordMetrics = dictionary.map(keyword => {
        let truePositives = [];
        let falsePositives = [];
        let totalPositives = 0;
        
        csvData.forEach(row => {
          const statement = (row[textColumn] || '').toLowerCase();
          const containsKeyword = statement.includes(keyword.toLowerCase());
          const groundTruth = parseInt(row[groundTruthColumn]);
          
          if (groundTruth === 1) {
            totalPositives++;
            if (containsKeyword) {
              truePositives.push(row);
            }
          } else if (groundTruth === 0 && containsKeyword) {
            falsePositives.push(row);
          }
        });
        
        const recall = totalPositives > 0 ? truePositives.length / totalPositives : 0;
        const precision = (truePositives.length + falsePositives.length) > 0 ? 
          truePositives.length / (truePositives.length + falsePositives.length) : 0;
        const f1 = recall > 0 && precision > 0 ? 
          2 * recall * precision / (recall + precision) : 0;
        
        return {
          keyword,
          truePositivesCount: truePositives.length,
          falsePositivesCount: falsePositives.length,
          recall: recall * 100,
          precision: precision * 100,
          f1Score: f1 * 100,
          truePositiveExamples: truePositives.slice(0, 3),
          falsePositiveExamples: falsePositives.slice(0, 3)
        };
      });
      
      // Sort by each metric and take top 10
      const byRecall = [...keywordMetrics]
        .sort((a, b) => b.recall - a.recall)
        .slice(0, 10);
      
      const byPrecision = [...keywordMetrics]
        .sort((a, b) => b.precision - a.precision)
        .slice(0, 10);
      
      const byF1 = [...keywordMetrics]
        .sort((a, b) => b.f1Score - a.f1Score)
        .slice(0, 10);
      
      setKeywordAnalysis({ byRecall, byPrecision, byF1 });
    } catch (err) {
      setError('Error analyzing keywords: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const loadSampleData = () => {
    parseCsv(sampleCsv);
  };

  // Convert keyword analysis to CSV
  const keywordAnalysisToCSV = () => {
    const currentData = activeMetric === 'recall' ? keywordAnalysis.byRecall :
                       activeMetric === 'precision' ? keywordAnalysis.byPrecision :
                       keywordAnalysis.byF1;
    
    if (!currentData || currentData.length === 0) return '';
    
    // CSV Header with proper Excel formatting
    let csv = 'Rank,Keyword,Recall_Percentage,Precision_Percentage,F1_Score_Percentage,True_Positives_Count,False_Positives_Count,True_Positive_Examples,False_Positive_Examples\n';
    
    // Add data rows
    currentData.forEach((item, index) => {
      // Properly escape quotes and handle special characters
      const keyword = `"${item.keyword.replace(/"/g, '""')}"`;
      
      // Format examples for Excel - join with semicolons and wrap in quotes
      const tpExamples = item.truePositiveExamples.length > 0 ? 
        `"${item.truePositiveExamples.map(ex => ex[textColumn]?.replace(/"/g, '""').replace(/\n/g, ' ')).join('; ')}"` : 
        '""';
      
      const fpExamples = item.falsePositiveExamples.length > 0 ? 
        `"${item.falsePositiveExamples.map(ex => ex[textColumn]?.replace(/"/g, '""').replace(/\n/g, ' ')).join('; ')}"` : 
        '""';
      
      csv += `${index + 1},${keyword},${item.recall.toFixed(2)},${item.precision.toFixed(2)},${item.f1Score.toFixed(2)},${item.truePositivesCount},${item.falsePositivesCount},${tpExamples},${fpExamples}\n`;
    });
    
    return csv;
  };

  // Convert classification results to CSV
  const classificationResultsToCSV = () => {
    if (!classificationResults || classificationResults.length === 0) return '';
    
    // CSV Header
    let csv = 'Row_ID,';
    if (csvHeaders.length > 0) {
      csv += csvHeaders.map(h => h.replace(/,/g, '_')).join(',') + ',';
    }
    csv += 'Predicted_Classification,Ground_Truth,Classification_Category,Matched_Keywords,Keyword_Count\n';
    
    // Add data rows
    classificationResults.forEach((result, index) => {
      let row = `${index + 1},`;
      
      // Add original CSV data
      if (csvHeaders.length > 0) {
        csvHeaders.forEach(header => {
          const value = result[header] || '';
          const escapedValue = `"${value.toString().replace(/"/g, '""').replace(/\n/g, ' ')}"`;
          row += escapedValue + ',';
        });
      }
      
      // Add classification results
      const matchedKeywords = result.matchedKeywords ? result.matchedKeywords.join('; ') : '';
      row += `${result.predicted},${result.groundTruth || ''},${result.category || ''},"${matchedKeywords}",${result.score || 0}\n`;
      
      csv += row;
    });
    
    return csv;
  };

  // Convert overall metrics to CSV
  const overallMetricsToCSV = () => {
    if (!overallMetrics) return '';
    
    let csv = 'Metric,Value,Count\n';
    csv += `Accuracy,${overallMetrics.accuracy}%,\n`;
    csv += `Precision,${overallMetrics.precision}%,\n`;
    csv += `Recall,${overallMetrics.recall}%,\n`;
    csv += `F1_Score,${overallMetrics.f1Score}%,\n`;
    csv += `True_Positives,,${overallMetrics.truePositives}\n`;
    csv += `False_Positives,,${overallMetrics.falsePositives}\n`;
    csv += `False_Negatives,,${overallMetrics.falseNegatives}\n`;
    csv += `True_Negatives,,${overallMetrics.trueNegatives}\n`;
    
    return csv;
  };

  const downloadCSV = (csv, filename) => {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const copyKeywordAnalysisAsCSV = async () => {
    const csv = keywordAnalysisToCSV();
    if (!csv) {
      setError('No keyword analysis data available to export');
      return;
    }
    
    const filename = `keyword-analysis-${activeMetric}-${new Date().toISOString().split('T')[0]}.csv`;
    
    // Try clipboard first, fallback to download
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(csv);
        // Show success message briefly
        const originalError = error;
        setError('✅ Keyword analysis copied to clipboard as CSV!');
        setTimeout(() => setError(originalError), 3000);
        return;
      } catch (clipboardErr) {
        console.log('Clipboard failed, falling back to download:', clipboardErr);
      }
    }
    
    // Fallback to download
    try {
      downloadCSV(csv, filename);
      const originalError = error;
      setError('✅ CSV file downloaded successfully!');
      setTimeout(() => setError(originalError), 3000);
    } catch (downloadErr) {
      console.error('Download failed:', downloadErr);
      setError('Failed to export CSV. Please try again.');
    }
  };

  const exportClassificationResults = async () => {
    const csv = classificationResultsToCSV();
    if (!csv) {
      setError('No classification results available to export');
      return;
    }
    
    const filename = `classification-results-${new Date().toISOString().split('T')[0]}.csv`;
    
    // Try clipboard first, fallback to download
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(csv);
        const originalError = error;
        setError('✅ Classification results copied to clipboard as CSV!');
        setTimeout(() => setError(originalError), 3000);
        return;
      } catch (clipboardErr) {
        console.log('Clipboard failed, falling back to download:', clipboardErr);
      }
    }
    
    // Fallback to download
    try {
      downloadCSV(csv, filename);
      const originalError = error;
      setError('✅ CSV file downloaded successfully!');
      setTimeout(() => setError(originalError), 3000);
    } catch (downloadErr) {
      console.error('Download failed:', downloadErr);
      setError('Failed to export CSV. Please try again.');
    }
  };

  const exportOverallMetrics = async () => {
    const csv = overallMetricsToCSV();
    if (!csv) {
      setError('No metrics available to export');
      return;
    }
    
    const filename = `overall-metrics-${new Date().toISOString().split('T')[0]}.csv`;
    
    // Try clipboard first, fallback to download
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(csv);
        const originalError = error;
        setError('✅ Overall metrics copied to clipboard as CSV!');
        setTimeout(() => setError(originalError), 3000);
        return;
      } catch (clipboardErr) {
        console.log('Clipboard failed, falling back to download:', clipboardErr);
      }
    }
    
    // Fallback to download
    try {
      downloadCSV(csv, filename);
      const originalError = error;
      setError('✅ CSV file downloaded successfully!');
      setTimeout(() => setError(originalError), 3000);
    } catch (downloadErr) {
      console.error('Download failed:', downloadErr);
      setError('Failed to export CSV. Please try again.');
    }
  };

  // Get false positives and false negatives
  const getFalsePositives = () => classificationResults.filter(r => r.category === 'FP');
  const getFalseNegatives = () => classificationResults.filter(r => r.category === 'FN');

  const renderKeywordTable = (keywords) => (
    <div className="space-y-6">
      {keywords.map((analysis, index) => (
        <div key={index} className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <span className="text-lg font-semibold">#{index + 1}</span>
              <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full font-medium">
                {analysis.keyword}
              </span>
            </div>
            <div className="flex gap-6 text-sm">
              <div>
                <span className="text-gray-600">Recall:</span>
                <span className="ml-2 font-medium text-green-600">{analysis.recall.toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-gray-600">Precision:</span>
                <span className="ml-2 font-medium text-blue-600">{analysis.precision.toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-gray-600">F1:</span>
                <span className="ml-2 font-medium text-purple-600">{analysis.f1Score.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-4">
            {/* True Positives */}
            <div className="bg-green-50 p-3 rounded-md">
              <h4 className="text-sm font-semibold text-green-800 mb-2">
                True Positives ({analysis.truePositivesCount})
              </h4>
              {analysis.truePositiveExamples.length > 0 ? (
                <ul className="space-y-2 text-xs">
                  {analysis.truePositiveExamples.map((example, i) => (
                    <li key={i} className="text-gray-700">
                      {example[textColumn]}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-500">No examples</p>
              )}
            </div>
            
            {/* False Positives */}
            <div className="bg-red-50 p-3 rounded-md">
              <h4 className="text-sm font-semibold text-red-800 mb-2">
                False Positives ({analysis.falsePositivesCount})
              </h4>
              {analysis.falsePositiveExamples.length > 0 ? (
                <ul className="space-y-2 text-xs">
                  {analysis.falsePositiveExamples.map((example, i) => (
                    <li key={i} className="text-gray-700">
                      {example[textColumn]}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-500">No examples</p>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-green-800 mb-2">Dictionary Classification Bot</h1>
        <p className="text-green-700">Enter keywords and classify statements to analyze their effectiveness</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md flex items-center">
          <AlertCircle className="h-5 w-5 mr-2" />
          {error}
        </div>
      )}

      {/* Step 1: Input CSV Data */}
      <div className="bg-white rounded-lg shadow-lg border-l-4 border-green-700 p-6">
        <h2 className="text-xl font-semibold mb-4 text-green-800">Step 1: Input Sample Data (CSV Format)</h2>
        
        <div className="space-y-4">
          {/* File Upload */}
          <div className="flex items-center gap-4">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-md flex items-center font-medium"
            >
              <FileUp className="h-5 w-5 mr-2" />
              Upload CSV File
            </button>
          </div>

          {/* Manual Input */}
          <textarea
            onChange={handleCsvInput}
            placeholder="Paste your CSV data here..."
            className="w-full p-3 border-2 border-green-200 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500 font-mono text-sm"
            rows="6"
          />

          {/* Column Selection */}
          {csvHeaders.length > 0 && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-green-50 rounded-md border border-green-200">
              <div>
                <label className="block text-sm font-medium text-green-800 mb-2">
                  Text Column for Analysis
                </label>
                <select
                  value={textColumn}
                  onChange={(e) => setTextColumn(e.target.value)}
                  className="w-full p-2 border-2 border-green-200 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
                >
                  <option value="">Select column...</option>
                  {csvHeaders.map(header => (
                    <option key={header} value={header}>{header}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-green-800 mb-2">
                  Ground Truth Column (0/1 values)
                </label>
                <select
                  value={groundTruthColumn}
                  onChange={(e) => setGroundTruthColumn(e.target.value)}
                  className="w-full p-2 border-2 border-green-200 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
                >
                  <option value="">Select column (optional)...</option>
                  {csvHeaders.map(header => (
                    <option key={header} value={header}>{header}</option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {csvData.length > 0 && (
            <p className="text-sm text-green-600 flex items-center">
              <CheckCircle2 className="h-4 w-4 mr-1" />
              Loaded {csvData.length} rows
            </p>
          )}
        </div>
      </div>

      {/* Step 2: Enter Dictionary */}
      <div className="bg-white rounded-lg shadow-lg border-l-4 border-yellow-500 p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-green-800">Step 2: Enter Keyword Dictionary</h2>
          {!isEditingDictionary && dictionary.length > 0 && (
            <button
              onClick={() => {
                setIsEditingDictionary(true);
                setDictionaryText(dictionary.map(k => `"${k}"`).join(','));
              }}
              className="text-green-700 hover:text-green-900 flex items-center"
            >
              <Edit3 className="h-4 w-4 mr-1" />
              Edit
            </button>
          )}
          {isEditingDictionary && dictionaryText && (
            <button
              onClick={saveDictionary}
              className="bg-green-700 text-white px-4 py-1 rounded-md hover:bg-green-800 flex items-center"
            >
              <Save className="h-4 w-4 mr-1" />
              Save
            </button>
          )}
        </div>
        
        {!isEditingDictionary && dictionary.length > 0 ? (
          <div className="bg-green-50 p-4 rounded-md border border-green-200">
            <p className="text-sm text-green-800 mb-2">Keywords ({dictionary.length}):</p>
            <div className="flex flex-wrap gap-2">
              {dictionary.map((keyword, index) => (
                <span key={index} className="bg-yellow-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div>
            <p className="text-sm text-green-700 mb-2">
              Enter keywords in format: "custom","customized","customization" or simply: custom, customized, customization
            </p>
            <textarea
              value={dictionaryText}
              onChange={(e) => setDictionaryText(e.target.value)}
              className="w-full p-3 border-2 border-green-200 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500 font-mono text-sm"
              rows="6"
              placeholder='"keyword1","keyword2","keyword3"'
            />
          </div>
        )}
      </div>

      {/* Step 3: Classify */}
      {dictionary.length > 0 && csvData.length > 0 && textColumn && (
        <div className="bg-white rounded-lg shadow-lg border-l-4 border-green-700 p-6">
          <h2 className="text-xl font-semibold mb-4 text-green-800">Step 3: Classify Statements</h2>
          <button
            onClick={classifyStatements}
            disabled={isClassifying}
            className="bg-green-700 text-white px-6 py-2 rounded-md hover:bg-green-800 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center font-medium"
          >
            {isClassifying ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Classifying...
              </>
            ) : (
              <>
                Classify Statements
              </>
            )}
          </button>
        </div>
      )}

      {/* Overall Classification Summary */}
      {overallMetrics && (
        <div className="bg-white rounded-lg shadow-lg border-l-4 border-yellow-500 p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-green-800">Classification Results Summary</h2>
            <div className="flex gap-2">
              <button
                onClick={exportOverallMetrics}
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md flex items-center text-sm font-medium"
              >
                <Copy className="h-4 w-4 mr-2" />
                Export Metrics CSV
              </button>
              <button
                onClick={exportClassificationResults}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md flex items-center text-sm font-medium"
              >
                <Copy className="h-4 w-4 mr-2" />
                Export Full Results CSV
              </button>
            </div>
          </div>
          
          {/* Metrics Display */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-green-50 p-4 rounded-lg text-center border-2 border-green-200">
              <p className="text-sm text-green-700 font-medium">Accuracy</p>
              <p className="text-2xl font-bold text-green-800">{overallMetrics.accuracy}%</p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg text-center border-2 border-yellow-200">
              <p className="text-sm text-yellow-700 font-medium">Precision</p>
              <p className="text-2xl font-bold text-yellow-800">{overallMetrics.precision}%</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg text-center border-2 border-green-300">
              <p className="text-sm text-green-700 font-medium">Recall</p>
              <p className="text-2xl font-bold text-green-900">{overallMetrics.recall}%</p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg text-center border-2 border-yellow-300">
              <p className="text-sm text-yellow-700 font-medium">F1 Score</p>
              <p className="text-2xl font-bold text-yellow-900">{overallMetrics.f1Score}%</p>
            </div>
          </div>

          {/* Confusion Matrix Summary */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>True Positives:</strong> {overallMetrics.truePositives} | 
              <strong> False Positives:</strong> {overallMetrics.falsePositives} | 
              <strong> False Negatives:</strong> {overallMetrics.falseNegatives} | 
              <strong> True Negatives:</strong> {overallMetrics.trueNegatives}
            </p>
          </div>

          {/* False Positives Section */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">False Positives (Incorrectly Classified as Positive)</h3>
            <div className="bg-red-50 p-4 rounded-lg">
              {getFalsePositives().length > 0 ? (
                <div className="space-y-3">
                  {(showAllFalsePositives ? getFalsePositives() : getFalsePositives().slice(0, 10)).map((result, index) => (
                    <div key={index} className="border-b border-red-200 pb-3 last:border-0">
                      <p className="text-sm text-gray-800 leading-relaxed">
                        {result[textColumn].split(' ').slice(0, 50).join(' ')}
                        {result[textColumn].split(' ').length > 50 && '...'}
                      </p>
                      <p className="text-xs text-red-600 mt-1">
                        Matched keywords: {result.matchedKeywords.join(', ')}
                      </p>
                    </div>
                  ))}
                  {!showAllFalsePositives && getFalsePositives().length > 10 && (
                    <button
                      onClick={() => setShowAllFalsePositives(true)}
                      className="text-red-600 hover:text-red-800 text-sm mt-2"
                    >
                      Show all {getFalsePositives().length} false positives
                    </button>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No false positives</p>
              )}
            </div>
          </div>

          {/* False Negatives Section */}
          <div>
            <h3 className="text-lg font-semibold mb-3">False Negatives (Missed Positive Cases)</h3>
            <div className="bg-yellow-50 p-4 rounded-lg">
              {getFalseNegatives().length > 0 ? (
                <div className="space-y-3">
                  {(showAllFalseNegatives ? getFalseNegatives() : getFalseNegatives().slice(0, 10)).map((result, index) => (
                    <div key={index} className="border-b border-yellow-200 pb-3 last:border-0">
                      <p className="text-sm text-gray-800 leading-relaxed">
                        {result[textColumn].split(' ').slice(0, 50).join(' ')}
                        {result[textColumn].split(' ').length > 50 && '...'}
                      </p>
                      <p className="text-xs text-yellow-700 mt-1">No keywords matched</p>
                    </div>
                  ))}
                  {!showAllFalseNegatives && getFalseNegatives().length > 10 && (
                    <button
                      onClick={() => setShowAllFalseNegatives(true)}
                      className="text-yellow-700 hover:text-yellow-800 text-sm mt-2"
                    >
                      Show all {getFalseNegatives().length} false negatives
                    </button>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No false negatives</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Step 4: Keyword Analysis */}
      {classificationResults.length > 0 && groundTruthColumn && (
        <div className="bg-white rounded-lg shadow-lg border-l-4 border-green-700 p-6">
          <h2 className="text-xl font-semibold mb-4 text-green-800">Step 4: Keyword Impact Analysis</h2>
          <p className="text-green-700 mb-4">
            Analyze keywords by different metrics to find the optimal set for your classification needs
          </p>
          <button
            onClick={analyzeKeywords}
            disabled={isAnalyzing}
            className="bg-yellow-500 text-white px-6 py-2 rounded-md hover:bg-yellow-600 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center mb-6 font-medium"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Analyzing Keywords...
              </>
            ) : (
              <>
                <BarChart className="h-5 w-5 mr-2" />
                Analyze Keyword Impact
              </>
            )}
          </button>
          
          {keywordAnalysis.byRecall.length > 0 && (
            <div>
              {/* Metric Tabs and Copy Button */}
              <div className="flex justify-between items-center mb-6">
                <div className="flex space-x-1 border-b">
                  <button
                    onClick={() => setActiveMetric('recall')}
                    className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                      activeMetric === 'recall' 
                        ? 'text-green-700 border-green-700' 
                        : 'text-gray-500 border-transparent hover:text-green-600'
                    }`}
                  >
                    <TrendingUp className="h-4 w-4 inline mr-2" />
                    Top by Recall
                  </button>
                  <button
                    onClick={() => setActiveMetric('precision')}
                    className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                      activeMetric === 'precision' 
                        ? 'text-yellow-600 border-yellow-600' 
                        : 'text-gray-500 border-transparent hover:text-yellow-600'
                    }`}
                  >
                    <Target className="h-4 w-4 inline mr-2" />
                    Top by Precision
                  </button>
                  <button
                    onClick={() => setActiveMetric('f1')}
                    className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                      activeMetric === 'f1' 
                        ? 'text-green-800 border-green-800' 
                        : 'text-gray-500 border-transparent hover:text-green-700'
                    }`}
                  >
                    <Activity className="h-4 w-4 inline mr-2" />
                    Top by F1 Score
                  </button>
                </div>
                <button
                  onClick={copyKeywordAnalysisAsCSV}
                  className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-md flex items-center text-sm font-medium"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Export Keyword Analysis CSV
                </button>
              </div>

              {/* Metric Description */}
              <div className="mb-6 p-4 bg-green-50 rounded-md border border-green-200">
                {activeMetric === 'recall' && (
                  <p className="text-sm text-green-800">
                    <strong>Recall:</strong> Percentage of true positive cases captured. High recall means the keyword catches most relevant statements.
                  </p>
                )}
                {activeMetric === 'precision' && (
                  <p className="text-sm text-green-800">
                    <strong>Precision:</strong> Percentage of predictions that are correct. High precision means the keyword rarely triggers on irrelevant statements.
                  </p>
                )}
                {activeMetric === 'f1' && (
                  <p className="text-sm text-green-800">
                    <strong>F1 Score:</strong> Harmonic mean of precision and recall. Balances both metrics for overall effectiveness.
                  </p>
                )}
              </div>

              {/* Keyword Analysis Table */}
              {activeMetric === 'recall' && renderKeywordTable(keywordAnalysis.byRecall)}
              {activeMetric === 'precision' && renderKeywordTable(keywordAnalysis.byPrecision)}
              {activeMetric === 'f1' && renderKeywordTable(keywordAnalysis.byF1)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
