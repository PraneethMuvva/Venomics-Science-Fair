// Simple JavaScript for Venomics ML Predictor

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const sequenceInput = document.getElementById('sequence');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');

    // Example sequence buttons
    document.querySelectorAll('.example-btn').forEach(button => {
        button.addEventListener('click', function() {
            sequenceInput.value = this.getAttribute('data-sequence');
        });
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const sequence = sequenceInput.value.trim().replace(/\s/g, '').toUpperCase();
        const topNValue = document.getElementById('top_n').value;
        const topN = topNValue === 'all' ? 'all' : parseInt(topNValue);

        if (!sequence) {
            showError('Please enter a protein sequence.');
            return;
        }

        if (!/^[ACDEFGHIKLMNPQRSTVWY]*$/i.test(sequence)) {
            showError('Invalid amino acids. Use only: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y');
            return;
        }

        if (sequence.length > 600) {
            showError('Sequence too long. Maximum 600 amino acids.');
            return;
        }

        try {
            setLoading(true);
            hideError();

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence, top_n: topN })
            });

            const data = await response.json();

            if (response.ok) {
                showResults(data);
            } else {
                showError(data.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Network error. Please try again.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(loading) {
        const btn = form.querySelector('button[type="submit"]');
        btn.textContent = loading ? 'Predicting...' : 'Predict Functions';
        btn.disabled = loading;
    }

    function showError(message) {
        document.getElementById('error-message').textContent = message;
        errorSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
    }

    function hideError() {
        errorSection.classList.add('d-none');
    }

    function showResults(data) {
        // Show sequence info
        document.getElementById('sequence-info').innerHTML = `
            <div class="sequence-display">
                <h6>Sequence (${data.sequence_length} amino acids):</h6>
                <code>${data.sequence}</code>
            </div>
        `;

        // Show predictions table
        let tableHTML = '<table class="table"><thead><tr><th>Rank</th><th>Function</th><th>Confidence</th></tr></thead><tbody>';

        data.predictions.forEach((pred, i) => {
            const color = pred.confidence >= 0.8 ? '#28a745' : pred.confidence >= 0.5 ? '#ffc107' : '#dc3545';
            tableHTML += `
                <tr>
                    <td>${i + 1}</td>
                    <td>
                        <div class="function-info">
                            <div class="go-term-header">
                                <span class="go-term">${pred.go_term}</span>
                                <span class="function-name">${pred.name}</span>
                            </div>
                            <small class="function-definition text-muted">${pred.definition}</small>
                        </div>
                    </td>
                    <td>
                        ${pred.confidence_percent}%
                        <div class="confidence-bar mt-1">
                            <div class="confidence-fill" style="width: ${pred.confidence_percent}%; background-color: ${color};"></div>
                        </div>
                    </td>
                </tr>
            `;
        });

        tableHTML += '</tbody></table>';
        document.getElementById('predictions-table').innerHTML = tableHTML;

        hideError();
        resultsSection.classList.remove('d-none');
    }
});