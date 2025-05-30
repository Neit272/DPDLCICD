import re
import json
import nltk
from typing import List, Dict, Set, Tuple
from pathlib import Path
import ast

class VulnerabilityPreprocessor:
    def __init__(self):
        # C/C++ keywords and standard types to preserve
        self.c_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
            'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
            'volatile', 'while', 'bool', 'true', 'false', 'NULL', 'nullptr'
        }
        
        # Standard library functions to preserve
        self.std_functions = {
            'printf', 'scanf', 'malloc', 'free', 'calloc', 'realloc', 'memcpy',
            'memset', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strncpy', 'strncat',
            'strncmp', 'sprintf', 'snprintf', 'fprintf', 'fopen', 'fclose',
            'fread', 'fwrite', 'system', 'exit', 'abort', 'gets', 'puts',
            'perror', 'ND_PRINT', 'nd_print', 'DBG_PRINT', 'dbg_print'
        }
        
        # Vulnerable API patterns for semantic slicing
        self.vulnerable_apis = {
            'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'system',
            'malloc', 'free', 'memcpy', 'strncpy', 'strncat'
        }
        
        # Logging context patterns
        self.logging_patterns = [
            r'\b(log|msg|error|print|warn|debug|info|trace|fatal)\w*\s*\(',
            r'\b(printf|fprintf|sprintf|snprintf)\s*\(',
            r'\b(cout|cerr|clog)\s*<<',
            r'\b(ND_PRINT|nd_print)\s*\(',           # Network debugging
            r'\b(DBG_PRINT|dbg_print)\s*\(',         # Debug printing
            r'\b(LOG_\w+|log_\w+)\s*\(',             # LOG_ERROR, LOG_INFO, etc.
            r'\b(PRINT_\w+|print_\w+)\s*\(',         # PRINT_DEBUG, etc.
            r'\b(TRACE_\w+|trace_\w+)\s*\(',         # TRACE_ENTER, etc.
            r'\b(DEBUG_\w+|debug_\w+)\s*\(',         # DEBUG_MSG, etc.
            r'\bperror\s*\(',                        # Standard error printing
            r'\bfprintf\s*\(\s*stderr',               # stderr printing
        ]
        
        # Regex for tokenization
        self.token_pattern = re.compile(r'''
            (?P<STRING>"(?:[^"\\]|\\.)*")|           # String literals
            (?P<CHAR>'(?:[^'\\]|\\.)*')|             # Character literals  
            (?P<NUMBER>0x[0-9a-fA-F]+|0[0-7]*|\d+\.?\d*[fFlL]?)|  # Numbers
            (?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*)|  # Identifiers
            (?P<OPERATOR><=|>=|==|!=|\+\+|--|&&|\|\||<<|>>|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)|  # Multi-char operators
            (?P<SINGLE>[{}()\[\];,.<>+\-*/&|^~!%=?:])| # Single char tokens
            (?P<WHITESPACE>\s+)|                     # Whitespace
            (?P<COMMENT>//.*?$|/\*.*?\*/)           # Comments
        ''', re.VERBOSE | re.MULTILINE | re.DOTALL)

    def remove_comments_and_preprocess(self, code: str) -> str:
        """Remove comments and basic cleanup"""
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def is_logging_context(self, code: str, string_pos: int) -> bool:
        """Check if string literal is in logging context"""
        # Look backwards from string position to find function call
        context_before = code[max(0, string_pos-150):string_pos]
        context_after = code[string_pos:string_pos+50]
        full_context = context_before + context_after
        
        # Original patterns
        logging_patterns = [
            r'\b(log|msg|error|print|warn|debug|info|trace|fatal)\w*\s*\(',
            r'\b(printf|fprintf|sprintf|snprintf)\s*\(',
            r'\b(cout|cerr|clog)\s*<<',
            r'\b(ND_PRINT|nd_print)\s*\(',           # Network debugging
            r'\b(DBG_PRINT|dbg_print)\s*\(',         # Debug printing  
            r'\b(LOG_\w+|log_\w+)\s*\(',             # LOG_ERROR, LOG_INFO, etc.
            r'\b(PRINT_\w+|print_\w+)\s*\(',         # PRINT_DEBUG, etc.
            r'\b(TRACE_\w+|trace_\w+)\s*\(',         # TRACE_ENTER, etc.
            r'\b(DEBUG_\w+|debug_\w+)\s*\(',         # DEBUG_MSG, etc.
            r'\bperror\s*\(',                        # Standard error printing
            r'\bfprintf\s*\(\s*stderr',               # stderr printing
        ]
        
        # After symbolic renaming patterns
        renamed_patterns = [
            r'\bFUNC_\d+.*printf\s*\(',              # Renamed function calling printf
            r'\bVAR_\d+.*printf\s*\(',               # Function pointer to printf
        ]
        
        # Check all patterns
        all_patterns = logging_patterns + renamed_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, full_context, re.IGNORECASE):
                return True
        
        # Check if immediately after printf( or similar
        immediate_patterns = [
            r'printf\s*\(\s*$',
            r'fprintf\s*\(\s*\w+\s*,\s*$',
            r'ND_PRINT\s*\(\s*$',
            r'DBG_PRINT\s*\(\s*$',
        ]
        
        for pattern in immediate_patterns:
            if re.search(pattern, context_before, re.IGNORECASE):
                return True
            
        return False

    def is_logic_context(self, code: str, string_pos: int) -> bool:
        """Check if string literal is in logic/conditional context"""
        # Look around the string for logic operators
        context = code[max(0, string_pos-30):string_pos+30]
        
        logic_patterns = [
            r'if\s*\(',
            r'while\s*\(',
            r'for\s*\(',
            r'switch\s*\(',
            r'==|!=|<=|>=|<|>',
            r'return\s+',
            r'&&|\|\|',
            r'case\s+',
            r'assert\s*\(',
        ]
        
        for pattern in logic_patterns:
            if re.search(pattern, context):
                return True
        return False

    def is_natural_language(self, text: str) -> bool:
        """Check if string contains natural language (improved heuristic)"""
        # Remove quotes
        content = text.strip('"\'')
        if len(content) < 3:
            return False
        
        # Skip format strings or technical strings
        if '%' in content and any(fmt in content for fmt in ['%s', '%d', '%c', '%x', '%p']):
            # This is likely a printf format string with natural language
            words = re.findall(r'\b[a-zA-Z]{2,}\b', content)
            return len(words) >= 2
        
        # Check for common natural language patterns
        words = re.findall(r'\b[a-zA-Z]{2,}\b', content)  # At least 2-char words
        if len(words) >= 2:  # At least 2 meaningful words
            return True
            
        # Check for common error message patterns
        error_patterns = [
            r'\berror\b', r'\binvalid\b', r'\bfailed\b', r'\bmemory\b',
            r'\bbuffer\b', r'\boverflow\b', r'\bundefined\b', r'\bdetected\b',
            r'\bprocessing\b', r'\binput\b', r'\buser\b'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
                
        return False

    def generalize_string_literals(self, code: str) -> str:
        """Step 1: Generalize string literals based on context"""
        result = []
        pos = 0
        
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"', code):
            # Add text before string
            result.append(code[pos:match.start()])
            
            string_literal = match.group()
            string_pos = match.start()
            
            # Check context
            if self.is_logic_context(code, string_pos):
                # Keep original in logic context
                result.append(string_literal)
            elif self.is_logging_context(code, string_pos):
                # Check if it's natural language
                if self.is_natural_language(string_literal):
                    result.append('"text"')
                else:
                    result.append(string_literal)
            else:
                # Default: check if natural language
                if self.is_natural_language(string_literal):
                    result.append('"text"')
                else:
                    result.append(string_literal)
            
            pos = match.end()
        
        result.append(code[pos:])
        return ''.join(result)

    def semantic_slicing(self, code: str, context_lines: int = 5) -> str:
        """Step 2: Optional semantic slicing around vulnerable APIs"""
        lines = code.split('\n')
        important_lines = set()
        
        for i, line in enumerate(lines):
            # Check for vulnerable APIs
            for api in self.vulnerable_apis:
                if re.search(rf'\b{api}\s*\(', line):
                    # Add current line and context
                    for j in range(max(0, i - context_lines), 
                                 min(len(lines), i + context_lines + 1)):
                        important_lines.add(j)
        
        if important_lines:
            # Return only important lines
            sliced_lines = [lines[i] for i in sorted(important_lines)]
            return '\n'.join(sliced_lines)
        else:
            # No vulnerable APIs found, return original
            return code

    def symbolic_renaming(self, code: str) -> str:
        """Step 3: Rename variables and functions symbolically"""
        # Extract function definitions first
        functions = []
        
        # Simple function pattern (can be improved with proper parsing)
        func_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^{]*\)\s*\{'
        
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            if func_name not in self.c_keywords and func_name not in self.std_functions:
                functions.append(func_name)
        
        # Create symbol mappings per function scope
        var_counter = 0
        func_counter = 0
        
        # Replace function names
        for func in functions:
            if func not in self.std_functions:
                code = re.sub(rf'\b{func}\b', f'FUNC_{func_counter}', code)
                func_counter += 1
        
        # Extract and replace variable names (simple heuristic)
        # This is a simplified version - a proper implementation would use AST parsing
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        seen_vars = set()
        def replace_var(match):
            nonlocal var_counter
            var_name = match.group(1)
            
            # Skip keywords, standard functions, and already renamed
            if (var_name in self.c_keywords or 
                var_name in self.std_functions or
                var_name.startswith(('VAR_', 'FUNC_')) or
                var_name in ['main']):  # Keep main function
                return var_name
            
            # Simple heuristic: if it looks like a variable (lowercase start or _)
            if (var_name[0].islower() or var_name.startswith('_')) and var_name not in seen_vars:
                seen_vars.add(var_name)
                replacement = f'VAR_{var_counter}'
                var_counter += 1
                return replacement
            
            return var_name
        
        # Apply variable renaming
        code = re.sub(identifier_pattern, replace_var, code)
        
        return code

    def tokenize_code(self, code: str) -> List[str]:
        """Step 4: Tokenize the code preserving C/C++ syntax"""
        tokens = []
        
        for match in self.token_pattern.finditer(code):
            token_type = match.lastgroup
            token_value = match.group()
            
            if token_type in ['WHITESPACE', 'COMMENT']:
                continue  # Skip whitespace and comments
            elif token_type == 'STRING':
                tokens.append(token_value)
            elif token_type == 'CHAR':
                tokens.append(token_value)
            elif token_type == 'NUMBER':
                # Keep numbers in logic contexts, might generalize others
                tokens.append(token_value)
            elif token_type == 'IDENTIFIER':
                tokens.append(token_value)
            elif token_type in ['OPERATOR', 'SINGLE']:
                tokens.append(token_value)
            else:
                tokens.append(token_value)
        
        return tokens

    def process_and_tokenize(self, code: str, apply_slicing: bool = False) -> List[str]:
        """Main pipeline function"""
        # Step 0: Basic cleanup
        code = self.remove_comments_and_preprocess(code)
        
        # Step 1: Optional semantic slicing (before renaming to preserve function names)
        if apply_slicing:
            code = self.semantic_slicing(code)
        
        # Step 2: Symbolic renaming FIRST
        code = self.symbolic_renaming(code)
        
        # Step 3: Generalize string literals AFTER renaming
        code = self.generalize_string_literals(code)
        
        # Step 4: Tokenization
        tokens = self.tokenize_code(code)
        
        return tokens

def process_dataset(input_path: str, output_path: str, apply_slicing: bool = False):
    """Process entire dataset"""
    preprocessor = VulnerabilityPreprocessor()
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
                original_code = data.get('func', '')
                
                # Process the code
                tokens = preprocessor.process_and_tokenize(original_code, apply_slicing)
                
                # Update data
                data['tokens'] = tokens
                data['processed_code'] = ' '.join(tokens)
                
                # Write processed data
                f_out.write(json.dumps(data) + '\n')
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

# Example usage
if __name__ == "__main__":
    # Test with a sample
    preprocessor = VulnerabilityPreprocessor()
    
    sample_code = '''
    int vulnerable_function(char* input) {
        char buffer[100];
        printf("Processing user input: %s", input);
        if (strlen(input) > 0) {
            strcpy(buffer, input);  // Vulnerable!
            printf("Buffer overflow detected at 0xdeadbeef");
            return 1;
        }
        return 0;
    }
    '''
    
    print("Original code:")
    print(sample_code)
    print("\nProcessed tokens:")
    tokens = preprocessor.process_and_tokenize(sample_code)
    print(tokens)
    print("\nProcessed code:")
    print(' '.join(tokens))