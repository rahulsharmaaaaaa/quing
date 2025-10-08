// Gemini AI Integration
// Using Google Gemini for better reasoning capabilities

export interface ExtractedQuestion {
  question_statement: string;
  question_type: 'MCQ' | 'MSQ' | 'NAT' | 'Subjective';
  options?: string[];
  answer?: string;
  solution?: string;
  question_number?: string;
  page_number?: number;
  has_image?: boolean;
  image_description?: string;
  is_continuation?: boolean;
  spans_multiple_pages?: boolean;
  uploaded_image?: string;
  topic_id?: string;
}

// Gemini API Keys - Round Robin System
let GEMINI_API_KEYS: string[] = [];
let currentKeyIndex = 0;

// Set API keys from user input
export function setGeminiApiKeys(keys: string[]) {
  GEMINI_API_KEYS = keys.filter(key => key.trim() !== '');
  currentKeyIndex = 0;
}

// Get next API key in round-robin fashion
function getNextGeminiKey(): string {
  if (GEMINI_API_KEYS.length === 0) {
    throw new Error('No Gemini API keys configured. Please add API keys first.');
  }
  
  const key = GEMINI_API_KEYS[currentKeyIndex];
  currentKeyIndex = (currentKeyIndex + 1) % GEMINI_API_KEYS.length;
  console.log(`Using Gemini API key index: ${currentKeyIndex === 0 ? GEMINI_API_KEYS.length - 1 : currentKeyIndex - 1}`);
  return key;
}

// Gemini API call function with retry logic
async function callGeminiAPI(prompt: string, imageBase64?: string, temperature: number = 0.1, maxTokens: number = 4000, retryCount: number = 0): Promise<string> {
  const apiKey = getNextGeminiKey();
  const maxRetries = 3;

  try {
    const requestBody: any = {
      contents: [{
        parts: []
      }],
      generationConfig: {
        temperature: temperature,
        maxOutputTokens: maxTokens,
      }
    };

    // Add text prompt
    requestBody.contents[0].parts.push({
      text: prompt
    });

    // Add image if provided
    if (imageBase64) {
      // Remove data URL prefix if present
      const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');
      requestBody.contents[0].parts.push({
        inline_data: {
          mime_type: "image/png",
          data: base64Data
        }
      });
    }

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-latest:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = errorText;

      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.error?.message || errorText;
      } catch (e) {
        // If parsing fails, use raw error text
      }

      // Handle rate limiting (429) or server errors (5xx)
      if ((response.status === 429 || response.status >= 500) && retryCount < maxRetries) {
        const waitTime = Math.pow(2, retryCount) * 2000; // Exponential backoff: 2s, 4s, 8s
        console.log(`Rate limit or server error. Retrying in ${waitTime/1000}s... (attempt ${retryCount + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        return callGeminiAPI(prompt, imageBase64, temperature, maxTokens, retryCount + 1);
      }

      throw new Error(`Gemini API error (${response.status}): ${errorMessage}`);
    }

    const data = await response.json();

    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
      // Handle content safety blocks
      if (data.promptFeedback?.blockReason) {
        throw new Error(`Content blocked: ${data.promptFeedback.blockReason}`);
      }
      throw new Error('Invalid response format from Gemini API');
    }

    return data.candidates[0].content.parts[0].text;
  } catch (error) {
    console.error('Gemini API call failed:', error);
    throw new Error(`Gemini API call failed: ${error.message}`);
  }
}

// Convert PDF to images using PDF.js
export async function convertPdfToImages(file: File): Promise<string[]> {
  const pdfjsLib = await import('pdfjs-dist');
  
  // Set worker source
  pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
  
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const images: string[] = [];
  
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const scale = 2.0; // Higher scale for better quality
    const viewport = page.getViewport({ scale });
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.height = viewport.height;
    canvas.width = viewport.width;
    
    await page.render({
      canvasContext: context,
      viewport: viewport
    }).promise;
    
    const imageDataUrl = canvas.toDataURL('image/png');
    images.push(imageDataUrl);
  }
  
  return images;
}

// Extract questions from PDF page using Gemini
export async function performExtraction(
  imageBase64: string, 
  pageNumber: number, 
  previousContext: string = '',
  pageMemory: Map<number, string> = new Map()
): Promise<ExtractedQuestion[]> {
  
  const contextInfo = previousContext ? `\n\nPrevious page context: ${previousContext.slice(-500)}` : '';
  const memoryInfo = pageMemory.size > 0 ? 
    `\n\nPage memory: ${Array.from(pageMemory.entries()).map(([p, content]) => `Page ${p}: ${content.slice(0, 200)}`).join('\n')}` : '';

  const prompt = `You are an expert at extracting questions from academic exam papers. Analyze this page image and extract ALL questions with perfect accuracy.

CRITICAL INSTRUCTIONS:
1. Extract EVERY question from this page, no matter how small or partial
2. For each question, determine the correct question type: MCQ, MSQ, NAT, or Subjective
3. Extract all options exactly as written (including mathematical notation)
4. DO NOT guess answers - only extract if clearly visible
5. Handle multi-page questions by noting continuation
6. Preserve all mathematical expressions, formulas, and special notation

Question Type Guidelines:
- MCQ: Single correct answer from multiple options
- MSQ: Multiple correct answers possible from options  
- NAT: Numerical answer type (no options, answer is a number)
- Subjective: Descriptive/essay type questions

${contextInfo}${memoryInfo}

Return a JSON array of questions in this exact format:
[
  {
    "question_statement": "Complete question text with all mathematical notation",
    "question_type": "MCQ|MSQ|NAT|Subjective",
    "options": ["Option A text", "Option B text", ...] or null for NAT/Subjective,
    "question_number": "Question number if visible",
    "has_image": true/false,
    "image_description": "Description of any diagrams/figures",
    "is_continuation": true/false,
    "spans_multiple_pages": true/false
  }
]

If no questions are found, return an empty array [].
Focus on accuracy and completeness. Extract everything visible.`;

  try {
    const response = await callGeminiAPI(prompt, imageBase64, 0.1, 4000);
    
    // Store page content in memory
    pageMemory.set(pageNumber, response.slice(0, 1000));
    
    // Parse JSON response
    let questions: ExtractedQuestion[] = [];
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        questions = JSON.parse(jsonMatch[0]);
      } else {
        console.warn('No JSON array found in response');
        return [];
      }
    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.log('Raw response:', response);
      return [];
    }

    // Add page number to each question
    questions = questions.map(q => ({
      ...q,
      page_number: pageNumber
    }));

    console.log(`Extracted ${questions.length} questions from page ${pageNumber}`);
    return questions;

  } catch (error) {
    console.error(`Error extracting questions from page ${pageNumber}:`, error);
    throw new Error(`Failed to extract questions from page ${pageNumber}: ${error.message}`);
  }
}

// Generate new questions for a topic using Gemini
export async function generateQuestionsForTopic(
  topic: any,
  examName: string,
  courseName: string,
  questionType: 'MCQ' | 'MSQ' | 'NAT' | 'Subjective',
  pyqs: any[],
  existingQuestionsContext: string,
  recentQuestions: string[],
  count: number = 1,
  topicNotes: string = ''
): Promise<ExtractedQuestion[]> {

  const pyqContext = pyqs.length > 0 ?
    `\n\nPREVIOUS YEAR QUESTIONS FOR INSPIRATION (Study these patterns carefully):\n${pyqs.map((q, i) =>
      `\nPYQ ${i+1} (${q.year} - ${q.slot || 'N/A'}):\nQuestion: ${q.question_statement}${q.options ? `\nOptions: ${q.options.join(', ')}` : ''}${q.answer ? `\nAnswer: ${q.answer}` : ''}${q.solution ? `\nSolution Approach: ${q.solution.slice(0, 200)}` : ''}`
    ).join('\n')}` : '';

  const existingContext = existingQuestionsContext ?
    `\n\nALREADY GENERATED QUESTIONS (Avoid duplication, create fresh questions):\n${existingQuestionsContext.slice(-1500)}` : '';

  const recentContext = recentQuestions.length > 0 ?
    `\n\nRECENTLY GENERATED (Must be different from these):\n${recentQuestions.slice(-3).join('\n')}` : '';

  const notesContext = topicNotes ?
    `\n\nTOPIC NOTES (Use these methods/concepts for the solution):\n${topicNotes.slice(0, 2000)}` : '';

  const prompt = `You are a professor creating ${examName} - ${courseName} questions. Generate ${count} unique ${questionType} question(s) for: "${topic.name}".

TOPIC INFORMATION:
- Topic: ${topic.name}
- Weightage: ${((topic.weightage || 0.02) * 100).toFixed(1)}%
${notesContext}
${pyqContext}
${existingContext}
${recentContext}

YOUR TASK:
1. Study the Previous Year Questions (PYQs) carefully - understand the pattern, difficulty level, and style
2. Create a NEW question that follows the same pattern but is completely fresh (DO NOT COPY)
3. When generating the solution, strictly use the methods/concepts from the Topic Notes
4. Ensure the question tests deep conceptual understanding

CRITICAL REQUIREMENTS for ${questionType} questions:

${questionType === 'MCQ' ? `
MCQ Requirements:
- Create exactly 4 options (A, B, C, D)
- EXACTLY ONE option must be correct
- Other 3 options must be plausible but incorrect
- Question must test conceptual understanding
- Provide clear, unambiguous correct answer
` : ''}

${questionType === 'MSQ' ? `
MSQ Requirements:
- Create exactly 4 options (A, B, C, D)
- 2-3 options should be correct (never just 1 or all 4)
- Incorrect options must be plausible distractors
- Question should test multiple concepts
- Clearly identify all correct options
` : ''}

${questionType === 'NAT' ? `
NAT Requirements:
- Question must have a numerical answer
- Answer should be a specific number (integer or decimal)
- No options needed
- Include proper units if applicable
- Ensure answer is calculable and unique
` : ''}

${questionType === 'Subjective' ? `
Subjective Requirements:
- Create comprehensive descriptive question
- Should test deep understanding
- Include multiple parts if appropriate
- Provide detailed solution approach
- No options needed
` : ''}

QUALITY STANDARDS:
1. Questions must be 100% original - inspired by PYQs but never duplicates
2. Use proper academic terminology and mathematical notation
3. Solutions MUST use methods from Topic Notes (not alternative approaches)
4. Match the ${examName} difficulty level and exam pattern
5. Write like a professor, not like AI (natural, clear, educational)
6. Avoid repetitive patterns - each question should feel unique
7. If you get stuck on a solution, try a different approach rather than continuing with errors

IMPORTANT: Avoid infinite loops in solution generation. If a solution has mistakes:
- Don't keep trying the same failed approach
- Verify your answer is mathematically/logically correct
- Use a completely different method if needed
- Double-check all calculations before finalizing

Return response in this exact JSON format:
[
  {
    "question_statement": "Complete question with natural wording (not robotic)",
    "question_type": "${questionType}",
    ${questionType === 'MCQ' || questionType === 'MSQ' ? '"options": ["Option A", "Option B", "Option C", "Option D"],' : '"options": null,'}
    "answer": "${questionType === 'MCQ' ? 'A' : questionType === 'MSQ' ? 'A, C' : questionType === 'NAT' ? '42.5' : 'Detailed answer'}",
    "solution": "Clear, step-by-step solution using Topic Notes methods. Write as a professor would explain to a student - conversational but precise."
  }
]

Generate exactly ${count} question(s) with verified accuracy.`;

  try {
    const response = await callGeminiAPI(prompt, undefined, 0.3, 3000);

    // Parse JSON response
    let questions: ExtractedQuestion[] = [];
    try {
      // Try to find JSON array in response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        questions = JSON.parse(jsonMatch[0]);
      } else {
        console.error('No JSON array found in response. Raw response:', response.slice(0, 500));
        throw new Error('No JSON array found in response');
      }

      // Validate questions structure
      if (!Array.isArray(questions) || questions.length === 0) {
        throw new Error('Invalid questions array returned');
      }

      // Validate each question has required fields
      for (const q of questions) {
        if (!q.question_statement || !q.question_type || !q.answer) {
          throw new Error('Question missing required fields (question_statement, question_type, or answer)');
        }
      }

    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.log('Raw response (first 1000 chars):', response.slice(0, 1000));
      throw new Error(`Failed to parse generated questions: ${parseError.message}`);
    }

    // Add topic_id to each question
    questions = questions.map(q => ({
      ...q,
      topic_id: topic.id
    }));

    console.log(`Generated ${questions.length} ${questionType} questions for topic: ${topic.name}`);
    return questions;

  } catch (error) {
    console.error(`Error generating questions for topic ${topic.name}:`, error);
    throw new Error(`Failed to generate questions: ${error.message}`);
  }
}

// Generate solutions for PYQs using Gemini
export async function generateSolutionsForPYQs(
  pyqs: any[],
  topicNotes: string = ''
): Promise<{ answer: string; solution: string }[]> {

  if (pyqs.length === 0) return [];

  const notesContext = topicNotes ?
    `\n\nTOPIC NOTES (Use ONLY these methods to solve):\n${topicNotes.slice(0, 2500)}` : '';

  const prompt = `You are a professor solving ${pyqs[0].topics?.name || 'academic'} questions. Provide accurate answers and clear solutions.
${notesContext}

CRITICAL INSTRUCTIONS:
1. When solving, STRICTLY use methods and concepts from the Topic Notes above
2. Do NOT use alternative methods or shortcuts not mentioned in the notes
3. If you encounter calculation errors, stop and recalculate - don't continue with wrong values
4. Write naturally as a professor would explain to a student, not in a robotic AI style
5. Verify your final answer before submitting

Questions to solve:
${pyqs.map((q, i) => `
Question ${i+1}:
${q.question_statement}
Type: ${q.question_type}
${q.options ? `Options:\n${q.options.map((opt, idx) => `  ${String.fromCharCode(65+idx)}. ${opt}`).join('\n')}` : ''}
`).join('\n')}

For each question provide:
- Correct answer (MCQ: 'A', MSQ: 'A, C', NAT: numerical value)
- Step-by-step solution using Topic Notes methods
- Clear explanation in professor-like natural language

Return response in this exact JSON format:
[
  {
    "answer": "Correct answer",
    "solution": "Natural, conversational step-by-step solution as a professor would explain it, using methods from Topic Notes"
  }
]

Verify all calculations and ensure solution correctness.`;

  try {
    const response = await callGeminiAPI(prompt, undefined, 0.1, 3000);
    
    // Parse JSON response
    let solutions: { answer: string; solution: string }[] = [];
    try {
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        solutions = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON array found in response');
      }
    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.log('Raw response:', response);
      throw new Error('Failed to parse generated solutions');
    }

    console.log(`Generated solutions for ${solutions.length} PYQs`);
    return solutions;

  } catch (error) {
    console.error('Error generating PYQ solutions:', error);
    throw new Error(`Failed to generate solutions: ${error.message}`);
  }
}

// Comprehensive question validation using Gemini AI
export async function validateQuestionWithGeminiAI(question: any): Promise<{ isWrong: boolean; reason: string }> {
  const prompt = `You are an expert question validator. Analyze this question thoroughly and determine if it's WRONG or CORRECT based on these strict criteria:

Question Details:
- Statement: ${question.question_statement}
- Type: ${question.question_type}
- Options: ${question.options ? question.options.join(', ') : 'None'}
- Provided Answer: ${question.answer || 'None'}

VALIDATION RULES:

For MCQ (Single Correct):
- WRONG if: No options are correct, multiple options are correct, or provided answer doesn't match any correct option
- CORRECT if: Exactly one option is correct and matches the provided answer

For MSQ (Multiple Correct):
- WRONG if: No options are correct, or provided answer doesn't include all correct options
- CORRECT if: One or more options are correct and provided answer matches all correct options

For NAT (Numerical Answer):
- WRONG if: Answer is not numerical, question is unsolvable, or provided answer is mathematically incorrect
- CORRECT if: Question is solvable and provided answer is mathematically correct

For Subjective:
- Always CORRECT (no validation needed)

ANALYSIS PROCESS:
1. Solve the question independently
2. Check if provided answer matches your solution
3. For MCQ/MSQ: Verify each option's correctness
4. For NAT: Verify numerical accuracy

Return response in this exact JSON format:
{
  "isWrong": true/false,
  "reason": "Detailed explanation of why the question is wrong or correct",
  "correctAnswer": "What the correct answer should be (if different from provided)"
}

Be extremely thorough and accurate in your analysis.`;

  try {
    const response = await callGeminiAPI(prompt, undefined, 0.1, 2000);
    
    // Parse JSON response
    let validation: { isWrong: boolean; reason: string; correctAnswer?: string };
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        validation = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON object found in response');
      }
    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.log('Raw response:', response);
      // Fallback: assume question is correct if parsing fails
      return { isWrong: false, reason: 'Validation parsing failed - marked as correct by default' };
    }

    console.log(`Question validation result: ${validation.isWrong ? 'WRONG' : 'CORRECT'} - ${validation.reason}`);
    return validation;

  } catch (error) {
    console.error('Error validating question:', error);
    // Fallback: assume question is correct if validation fails
    return { isWrong: false, reason: `Validation failed: ${error.message} - marked as correct by default` };
  }
}

// Simple client-side validation (kept for backward compatibility)
export function validateQuestionAnswer(question: ExtractedQuestion): { isValid: boolean; reason: string } {
  if (!question.question_statement || question.question_statement.trim().length === 0) {
    return { isValid: false, reason: 'Empty question statement' };
  }

  if (!question.question_type || !['MCQ', 'MSQ', 'NAT', 'Subjective'].includes(question.question_type)) {
    return { isValid: false, reason: 'Invalid question type' };
  }

  // For MCQ and MSQ, options are required
  if ((question.question_type === 'MCQ' || question.question_type === 'MSQ') && 
      (!question.options || question.options.length === 0)) {
    return { isValid: false, reason: 'MCQ/MSQ questions require options' };
  }

  // Basic validation passed
  return { isValid: true, reason: 'Question passes basic validation' };
}

// Export the API key management functions for external use
export { getNextGeminiKey, callGeminiAPI };