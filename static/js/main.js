// Main JavaScript for PDF Information Extractor

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const filesInput = document.getElementById('files');
    const queryInput = document.getElementById('query');

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const files = filesInput.files;
        const query = queryInput.value.trim();
        
        if (files.length === 0) {
            showError('Please select at least one PDF file.');
            return;
        }
        
        if (!query) {
            showError('Please enter a query describing what information you want to extract.');
            return;
        }

        // Validate file types and sizes
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showError(`File "${file.name}" is not a PDF file.`);
                return;
            }
            
            if (file.size > 50 * 1024 * 1024) { // 50MB
                showError(`File "${file.name}" is too large. Maximum size is 50MB.`);
                return;
            }
        }

        processFiles(files, query);
    });

    // Handle example query clicks
    document.querySelectorAll('.list-unstyled li').forEach(function(item) {
        item.addEventListener('click', function() {
            const exampleText = this.textContent.replace(/.*"([^"]*)".*/, '$1');
            queryInput.value = exampleText;
            queryInput.focus();
        });
    });

    function processFiles(files, query) {
        // Show progress, hide other sections
        showProgress();
        hideResults();
        hideError();
        
        // Disable submit button
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';

        const formData = new FormData();
        
        // Add files
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }
        
        // Add query
        formData.append('query', query);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideProgress();
            
            if (data.error) {
                showError(data.error);
            } else if (data.success) {
                showResults(data.results);
            } else {
                showError('Unexpected response format.');
            }
        })
        .catch(error => {
            hideProgress();
            console.error('Error:', error);
            showError('An error occurred while processing your request. Please try again.');
        })
        .finally(() => {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Extract Information';
        });
    }

    function showProgress() {
        progressSection.style.display = 'block';
        progressSection.scrollIntoView({ behavior: 'smooth' });
    }

    function hideProgress() {
        progressSection.style.display = 'none';
    }

    function showResults(results) {
        const resultsContent = document.getElementById('resultsContent');
        
        if (results.error) {
            showError(results.error);
            return;
        }

        let html = '';
        
        // Main answer section
        if (results.answer) {
            html += `
                <div class="results-content fade-in">
                    <h5 class="mb-3">
                        <i class="fas fa-lightbulb me-2 text-warning"></i>
                        Answer to: "${results.query}"
                    </h5>
                    <div class="answer-text">
                        ${formatAnswer(results.answer)}
                    </div>
                </div>
            `;
        }

        // Metadata section
        if (results.metadata) {
            html += `
                <div class="metadata-section fade-in">
                    <h6 class="mb-3">
                        <i class="fas fa-info-circle me-2"></i>
                        Processing Summary
                    </h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="metadata-item">
                                <span class="metadata-label">Documents Processed:</span>
                                <span class="metadata-value">${results.metadata.input_documents ? results.metadata.input_documents.length : 0}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Sections Analyzed:</span>
                                <span class="metadata-value">${results.metadata.sections_analyzed || 0}</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="metadata-item">
                                <span class="metadata-label">Top Sections Used:</span>
                                <span class="metadata-value">${results.metadata.top_sections_used || 0}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Processing Time:</span>
                                <span class="metadata-value">${formatDateTime(results.metadata.processing_timestamp)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Detailed sections
        if (results.detailed_sections && results.detailed_sections.length > 0) {
            html += `
                <div class="mt-4">
                    <h6 class="mb-3">
                        <i class="fas fa-list me-2"></i>
                        Source Details
                    </h6>
            `;
            
            results.detailed_sections.forEach((section, index) => {
                const relevancePercentage = Math.round((section.relevance_score || 0) * 100);
                html += `
                    <div class="section-card fade-in" style="animation-delay: ${index * 0.1}s">
                        <div class="section-header">
                            <div>
                                <h6 class="mb-1">
                                    <i class="fas fa-file-pdf me-2 text-danger"></i>
                                    ${section.document}
                                </h6>
                                <small class="text-muted">
                                    ${section.section_title} • Page ${section.page_number}
                                </small>
                            </div>
                            <span class="section-score">${relevancePercentage}% relevant</span>
                        </div>
                        <div class="expandable-content" id="content-${index}">
                            <p class="mb-2"><strong>Summary:</strong></p>
                            <p>${section.summary || 'No summary available'}</p>
                            <div class="full-content" style="display: none;">
                                <p class="mb-2 mt-3"><strong>Full Content:</strong></p>
                                <p class="text-muted small">${section.full_content || 'No full content available'}</p>
                            </div>
                        </div>
                        <button class="expand-btn" onclick="toggleContent(${index})">
                            <i class="fas fa-chevron-down me-1"></i>
                            Show Full Content
                        </button>
                    </div>
                `;
            });
            
            html += '</div>';
        }

        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }

    function formatAnswer(answer) {
        // Split by document sections and format nicely
        const sections = answer.split('\n\n');
        let formatted = '';
        
        sections.forEach((section, index) => {
            if (section.trim()) {
                if (section.startsWith('From ')) {
                    // This is a source header
                    const parts = section.split(':\n');
                    if (parts.length >= 2) {
                        formatted += `
                            <div class="source-info">
                                <div class="source-title">${parts[0]}</div>
                                <p class="mb-0">${parts.slice(1).join(':\n')}</p>
                            </div>
                        `;
                    } else {
                        formatted += `<p>${section}</p>`;
                    }
                } else {
                    formatted += `<p>${section}</p>`;
                }
            }
        });
        
        return formatted;
    }

    function formatDateTime(isoString) {
        if (!isoString) return 'Unknown';
        const date = new Date(isoString);
        return date.toLocaleString();
    }

    // File input change handler to show file names
    filesInput.addEventListener('change', function() {
        const files = this.files;
        const fileNames = Array.from(files).map(file => file.name);
        
        if (fileNames.length > 0) {
            const fileList = fileNames.length > 3 
                ? `${fileNames.slice(0, 3).join(', ')} and ${fileNames.length - 3} more...`
                : fileNames.join(', ');
            
            // Update the form text to show selected files
            const existingText = document.querySelector('.file-selection-text');
            if (existingText) {
                existingText.remove();
            }
            
            const fileText = document.createElement('div');
            fileText.className = 'file-selection-text form-text text-success';
            fileText.innerHTML = `<i class="fas fa-check-circle me-1"></i>Selected: ${fileList}`;
            filesInput.parentNode.appendChild(fileText);
        }
    });
});

// Global function to toggle content visibility
function toggleContent(index) {
    const content = document.getElementById(`content-${index}`);
    const button = content.nextElementSibling;
    const fullContent = content.querySelector('.full-content');
    
    if (fullContent.style.display === 'none') {
        fullContent.style.display = 'block';
        content.classList.add('expanded');
        button.innerHTML = '<i class="fas fa-chevron-up me-1"></i>Show Less';
    } else {
        fullContent.style.display = 'none';
        content.classList.remove('expanded');
        button.innerHTML = '<i class="fas fa-chevron-down me-1"></i>Show Full Content';
    }
}