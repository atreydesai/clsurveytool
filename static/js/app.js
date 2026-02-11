/**
 * CL Survey Tool - Frontend Logic
 */

// State
let state = {
    entries: { pending: [], saved: [] },
    constants: {},
    activeTab: 'pending',
    selectedEntryId: null,
    debounceTimer: null,
    backgroundTaskCount: 0
};

// ============================================================================
// UI Helper Functions
// ============================================================================

function updateGlobalIndicator() {
    let indicator = document.getElementById('global-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'global-indicator';
        indicator.className = 'global-indicator';
        document.body.appendChild(indicator);
    }

    if (state.backgroundTaskCount > 0) {
        indicator.textContent = `Analyzing ${state.backgroundTaskCount} paper${state.backgroundTaskCount > 1 ? 's' : ''}...`;
        indicator.style.display = 'block';
    } else {
        indicator.style.display = 'none';
    }
}

function showNotification(message) {
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    container.appendChild(toast);

    // Animate in
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after 3s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================================
// API Functions
// ============================================================================

async function api(endpoint, options = {}) {
    const response = await fetch(`/api${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    return response.json();
}

async function loadEntries() {
    const sources = getSelectedSources();
    const params = sources ? `?sources=${sources}` : '';
    const data = await api(`/entries${params}`);
    state.entries = { pending: data.pending || [], saved: data.saved || [] };
    state.constants = data.constants || {};
    renderEntryList();
    updateCounts();
}

async function importBibtex(bibtex) {
    const result = await api('/import', {
        method: 'POST',
        body: JSON.stringify({ bibtex })
    });
    if (result.entries) {
        await loadEntries();
    }
    return result;
}

async function saveEntry(entryId, data) {
    const result = await api(`/entries/${entryId}`, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
    showSaveIndicator();
    return result;
}

async function commitEntry(entryId) {
    const result = await api(`/entries/${entryId}/commit`, { method: 'POST' });
    if (!result.error) {
        await loadEntries();
        state.selectedEntryId = null;
        renderEditor();
    }
    return result;
}

async function deleteEntry(entryId) {
    const result = await api(`/entries/${entryId}`, { method: 'DELETE' });
    if (!result.error) {
        await loadEntries();
        if (state.selectedEntryId === entryId) {
            state.selectedEntryId = null;
            renderEditor();
        }
    }
    return result;
}

async function runAnalysis(notes) {
    return await api('/analyze', {
        method: 'POST',
        body: JSON.stringify({ notes })
    });
}

async function loadStats() {
    return await api('/stats');
}

// ============================================================================
// Rendering Functions
// ============================================================================

function updateCounts() {
    document.getElementById('pending-count').textContent = state.entries.pending.length;
    document.getElementById('saved-count').textContent = state.entries.saved.length;
}

function renderEntryList() {
    const list = document.getElementById('entry-list');
    const entries = state.activeTab === 'pending' ? state.entries.pending : state.entries.saved;

    if (entries.length === 0) {
        list.innerHTML = `<div class="entry-item"><p style="color: var(--text-muted);">No entries</p></div>`;
        return;
    }

    list.innerHTML = entries.map(entry => `
        <div class="entry-item ${state.selectedEntryId === entry.id ? 'active' : ''}" 
             data-id="${entry.id}">
            <div class="entry-title">${escapeHtml(entry.title || 'Untitled')}</div>
            <div class="entry-meta">${entry.year || '?'} · ${entry.journal || 'Unknown'}</div>
        </div>
    `).join('');

    // Add click handlers
    list.querySelectorAll('.entry-item').forEach(item => {
        item.addEventListener('click', () => {
            state.selectedEntryId = item.dataset.id;
            renderEntryList();
            renderEditor();
        });
    });
}

function renderEditor() {
    const editor = document.getElementById('editor');

    if (!state.selectedEntryId) {
        editor.innerHTML = `<div class="editor-placeholder"><p>Select an entry to edit</p></div>`;
        return;
    }

    const entry = [...state.entries.pending, ...state.entries.saved]
        .find(e => e.id === state.selectedEntryId);

    if (!entry) {
        editor.innerHTML = `<div class="editor-placeholder"><p>Entry not found</p></div>`;
        return;
    }

    const isPending = state.entries.pending.some(e => e.id === entry.id);
    const searchString = `${entry.title || ''} ${(entry.authors || [])[0] || ''}`;

    // Ensure minimum affiliations: 1 by default, 2 if multiple authors
    const authorCount = (entry.authors || []).length;
    const minAffiliations = authorCount > 1 ? 2 : 1;
    if (!entry.affiliations || entry.affiliations.length < minAffiliations) {
        entry.affiliations = entry.affiliations || [];
        while (entry.affiliations.length < minAffiliations) {
            entry.affiliations.push({ university: '', country: '', discipline: '' });
        }
    }

    editor.innerHTML = `
        <div class="editor-form">
            <div class="form-header">
                <h2 class="form-title">Edit Entry</h2>
                <div class="form-actions">
                    ${isPending ? `<button class="btn btn-success" id="btn-commit">Commit to Dataset</button>` : ''}
                    <button class="btn btn-danger btn-small" id="btn-delete">Delete</button>
                </div>
            </div>
            
            <div class="editor-columns">
                <!-- Left Column: Metadata -->
                <div class="editor-col-left">
                    <div class="form-row">
                        <div class="form-group" style="flex: 3;">
                            <label>Title</label>
                            <input type="text" id="field-title" value="${escapeHtml(entry.title || '')}">
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label>Year</label>
                            <input type="text" id="field-year" value="${escapeHtml(entry.year || '')}">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Journal/Venue</label>
                        <input type="text" id="field-journal" value="${escapeHtml(entry.journal || '')}">
                    </div>
                    
                    <div class="form-group">
                        <label>Authors (one per line)</label>
                        <textarea id="field-authors" style="min-height: 80px;">${(entry.authors || []).join('\n')}</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>DOI</label>
                        <input type="text" id="field-doi" value="${escapeHtml(entry.doi || '')}" placeholder="e.g. 10.1000/example">
                    </div>
                    
                    <!-- Google Scholar Copy -->
                    <div class="form-group">
                        <label>Google Scholar Search</label>
                        <button class="copy-btn" id="btn-copy">📋 Copy: ${escapeHtml(searchString.substring(0, 40))}...</button>
                    </div>
                    
                    <!-- Analysis Notes -->
                    <div class="form-group">
                        <label>Analysis Notes</label>
                        <textarea id="field-notes" style="min-height: 150px;">${escapeHtml(entry.analysis_notes || '')}</textarea>
                        <button class="btn btn-primary" id="btn-analyze" style="margin-top: 0.5rem;">Run AI Analysis</button>
                    </div>
                </div>
                
                <!-- Right Column: Classifications -->
                <div class="editor-col-right">
                    <!-- Affiliations -->
                    <div class="form-group">
                        <label>Affiliations</label>
                        <table class="affiliations-table">
                            <thead>
                                <tr>
                                    <th>University</th>
                                    <th>Country</th>
                                    <th>Discipline</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody id="affiliations-body">
                                ${(entry.affiliations || []).map((aff, i) => `
                                    <tr data-index="${i}">
                                        <td class="autocomplete-cell">
                                            <input type="text" class="aff-university autocomplete-input" data-list="universities" value="${escapeHtml(aff.university || '')}" placeholder="Start typing...">
                                        </td>
                                        <td class="autocomplete-cell">
                                            <input type="text" class="aff-country autocomplete-input" data-list="countries" value="${escapeHtml(aff.country || '')}">
                                        </td>
                                        <td class="autocomplete-cell">
                                            <input type="text" class="aff-discipline autocomplete-input" data-list="disciplines" value="${escapeHtml(aff.discipline || '')}">
                                        </td>
                                        <td><button class="btn btn-danger btn-small aff-remove">×</button></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                        <button class="btn btn-small add-row-btn" id="btn-add-affiliation">+ Add Row</button>
                    </div>
                    
                    <!-- Species Categories -->
                    <div class="form-group">
                        <label>Species Categories</label>
                        <div class="dual-select" data-field="species_categories">
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Available</div>
                                <div class="dual-select-options available">
                                    ${(state.constants.species_categories || [])
            .filter(cat => !(entry.species_categories || []).includes(cat))
            .map(cat => `<button type="button" class="select-item" data-value="${escapeHtml(cat)}">${escapeHtml(cat)}</button>`)
            .join('')}
                                </div>
                            </div>
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Selected</div>
                                <div class="dual-select-options selected">
                                    ${(entry.species_categories || [])
            .map(cat => `<button type="button" class="select-item selected" data-value="${escapeHtml(cat)}">${escapeHtml(cat)}</button>`)
            .join('') || '<span class="empty-hint">Click to add</span>'}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Specialized Species -->
                    <div class="form-group">
                        <label>Specialized Species (comma-separated)</label>
                        <input type="text" id="field-specialized-species" value="${(entry.specialized_species || []).join(', ')}">
                    </div>
                    
                    <!-- Computational Stages -->
                    <div class="form-group">
                        <label>Computational Stages</label>
                        <div class="dual-select" data-field="computational_stages">
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Available</div>
                                <div class="dual-select-options available">
                                    ${(state.constants.computational_stages || [])
            .filter(stage => !(entry.computational_stages || []).includes(stage))
            .map(stage => `<button type="button" class="select-item" data-value="${escapeHtml(stage)}">${escapeHtml(stage)}</button>`)
            .join('')}
                                </div>
                            </div>
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Selected</div>
                                <div class="dual-select-options selected">
                                    ${(entry.computational_stages || [])
            .map(stage => `<button type="button" class="select-item selected" data-value="${escapeHtml(stage)}">${escapeHtml(stage)}</button>`)
            .join('') || '<span class="empty-hint">Click to add</span>'}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Linguistic Features -->
                    <div class="form-group">
                        <label>Linguistic Features</label>
                        <div class="dual-select dual-select-tall" data-field="linguistic_features">
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Available</div>
                                <div class="dual-select-options available">
                                    ${(state.constants.linguistic_features || [])
            .filter(feat => !(entry.linguistic_features || []).includes(feat))
            .map(feat => `<button type="button" class="select-item" data-value="${escapeHtml(feat)}">${escapeHtml(feat)}</button>`)
            .join('')}
                                </div>
                            </div>
                            <div class="dual-select-pane">
                                <div class="dual-select-header">Selected</div>
                                <div class="dual-select-options selected">
                                    ${(entry.linguistic_features || [])
            .map(feat => `<button type="button" class="select-item selected" data-value="${escapeHtml(feat)}">${escapeHtml(feat)}</button>`)
            .join('') || '<span class="empty-hint">Click to add</span>'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Attach event handlers
    attachEditorHandlers(entry);
}

function attachEditorHandlers(entry) {
    // Auto-save on input changes (debounced)
    const inputs = ['field-title', 'field-year', 'field-journal', 'field-authors', 'field-doi', 'field-notes', 'field-specialized-species'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => debounceAutoSave(entry));
        }
    });

    // Copy button
    document.getElementById('btn-copy')?.addEventListener('click', () => {
        const searchString = `${entry.title || ''} ${(entry.authors || [])[0] || ''}`;
        navigator.clipboard.writeText(searchString);
        document.getElementById('btn-copy').textContent = '✅ Copied!';
        setTimeout(() => renderEditor(), 1500);
    });

    // Commit button
    document.getElementById('btn-commit')?.addEventListener('click', async () => {
        await commitEntry(entry.id);
    });

    // Delete button
    document.getElementById('btn-delete')?.addEventListener('click', async () => {
        if (confirm('Delete this entry?')) {
            await deleteEntry(entry.id);
        }
    });

    // AI Analysis button
    document.getElementById('btn-analyze')?.addEventListener('click', async () => {
        const notes = document.getElementById('field-notes').value;
        if (!notes) {
            alert('Add analysis notes first');
            return;
        }
        const btn = document.getElementById('btn-analyze');
        btn.disabled = true;
        btn.textContent = 'Analyzing...';

        // Increment global task count
        state.backgroundTaskCount++;
        updateGlobalIndicator();

        try {
            const result = await runAnalysis(notes);
            if (result.error) {
                alert('Error: ' + result.error);
            } else {
                // Apply results
                entry.linguistic_features = result.linguistic_features || [];
                entry.species_categories = [...new Set(result.species_categories || [])];
                entry.specialized_species = result.specialized_species || [];
                entry.computational_stages = result.computational_stages || [];
                if (result.affiliations && result.affiliations.length > 0) {
                    entry.affiliations = result.affiliations;
                }
                await saveEntry(entry.id, entry);

                // If this is still the selected entry, re-render
                if (state.selectedEntryId === entry.id) {
                    renderEditor();
                } else {
                    // Otherwise notify user
                    showNotification(`Analysis complete for "${entry.title}"`);
                }
            }
        } catch (e) {
            console.error(e);
            alert('Analysis failed');
        } finally {
            state.backgroundTaskCount--;
            updateGlobalIndicator();

            // Only update button if it still exists (user hasn't navigated away)
            const currentBtn = document.getElementById('btn-analyze');
            if (currentBtn) {
                currentBtn.disabled = false;
                currentBtn.textContent = 'Run AI Analysis';
            }
        }
    });

    // Add affiliation row
    document.getElementById('btn-add-affiliation')?.addEventListener('click', () => {
        if (!entry.affiliations) entry.affiliations = [];
        entry.affiliations.push({ university: '', country: '', discipline: '' });
        renderEditor();
    });

    // Remove affiliation row
    document.querySelectorAll('.aff-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const row = e.target.closest('tr');
            const index = parseInt(row.dataset.index);
            entry.affiliations.splice(index, 1);
            collectFormData(entry);
            saveEntry(entry.id, entry);
            renderEditor();
        });
    });

    // Affiliation input changes with custom autocomplete
    document.querySelectorAll('.autocomplete-input').forEach(input => {
        const listType = input.dataset.list;
        let options = [];

        if (listType === 'universities') {
            // Combine known universities with current entry's universities
            const currentUnis = (entry.affiliations || []).map(a => a.university).filter(u => u);
            options = [...new Set([...(state.constants.known_universities || []), ...currentUnis])];
        } else if (listType === 'countries') {
            options = state.constants.countries || [];
        } else if (listType === 'disciplines') {
            options = state.constants.disciplines || [];
        }

        // Create dropdown container
        let dropdown = input.parentElement.querySelector('.autocomplete-dropdown');
        if (!dropdown) {
            dropdown = document.createElement('div');
            dropdown.className = 'autocomplete-dropdown';
            input.parentElement.appendChild(dropdown);
        }

        const showDropdown = () => {
            const query = input.value.toLowerCase();
            const filtered = options.filter(opt => opt.toLowerCase().includes(query));

            if (filtered.length === 0 || (filtered.length === 1 && filtered[0].toLowerCase() === query)) {
                dropdown.style.display = 'none';
                return;
            }

            dropdown.innerHTML = filtered.slice(0, 8).map((opt, i) => `
                <div class="autocomplete-option ${i === 0 ? 'first' : ''}">${escapeHtml(opt)}</div>
            `).join('');
            dropdown.style.display = 'block';

            dropdown.querySelectorAll('.autocomplete-option').forEach(opt => {
                opt.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    input.value = opt.textContent;
                    dropdown.style.display = 'none';

                    // Auto-fill country and discipline when selecting a pre-saved university
                    if (listType === 'universities') {
                        const universityCountryMap = state.constants.university_country_map || {};
                        const universityDisciplineMap = state.constants.university_discipline_map || {};
                        const selectedUniversity = opt.textContent;
                        const row = input.closest('tr');

                        if (universityCountryMap[selectedUniversity]) {
                            const countryInput = row.querySelector('.aff-country');
                            const associatedCountry = universityCountryMap[selectedUniversity];
                            if (countryInput && !countryInput.value) {
                                countryInput.value = associatedCountry;
                            }

                            // Auto-fill discipline using university|country key
                            const disciplineKey = `${selectedUniversity}|${associatedCountry}`;
                            if (universityDisciplineMap[disciplineKey]) {
                                const disciplineInput = row.querySelector('.aff-discipline');
                                if (disciplineInput && !disciplineInput.value) {
                                    disciplineInput.value = universityDisciplineMap[disciplineKey];
                                }
                            }
                        }
                    }

                    debounceAutoSave(entry);
                });
            });
        };

        input.addEventListener('input', () => {
            showDropdown();
            debounceAutoSave(entry);
        });

        input.addEventListener('focus', showDropdown);

        input.addEventListener('blur', () => {
            setTimeout(() => dropdown.style.display = 'none', 150);
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const firstOption = dropdown.querySelector('.autocomplete-option');
                if (firstOption && dropdown.style.display === 'block') {
                    input.value = firstOption.textContent;
                    dropdown.style.display = 'none';

                    // Auto-fill country and discipline when selecting a pre-saved university via Enter
                    if (listType === 'universities') {
                        const universityCountryMap = state.constants.university_country_map || {};
                        const universityDisciplineMap = state.constants.university_discipline_map || {};
                        const selectedUniversity = firstOption.textContent;
                        const row = input.closest('tr');

                        if (universityCountryMap[selectedUniversity]) {
                            const countryInput = row.querySelector('.aff-country');
                            const associatedCountry = universityCountryMap[selectedUniversity];
                            if (countryInput && !countryInput.value) {
                                countryInput.value = associatedCountry;
                            }

                            // Auto-fill discipline using university|country key
                            const disciplineKey = `${selectedUniversity}|${associatedCountry}`;
                            if (universityDisciplineMap[disciplineKey]) {
                                const disciplineInput = row.querySelector('.aff-discipline');
                                if (disciplineInput && !disciplineInput.value) {
                                    disciplineInput.value = universityDisciplineMap[disciplineKey];
                                }
                            }
                        }
                    }

                    debounceAutoSave(entry);
                }
            } else if (e.key === 'Escape') {
                dropdown.style.display = 'none';
            }
        });
    });

    // Dual-select handlers for species, stages, and features
    document.querySelectorAll('.dual-select .select-item').forEach(btn => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.value;
            const field = btn.closest('.dual-select').dataset.field;

            if (!entry[field]) entry[field] = [];

            if (entry[field].includes(value)) {
                // Remove from selected
                entry[field] = entry[field].filter(v => v !== value);
            } else {
                // Add to selected
                entry[field].push(value);
            }

            saveEntry(entry.id, { [field]: entry[field] });
            renderEditor();  // Re-render for smooth transition
        });
    });
}

function collectFormData(entry) {
    // Collect all form data into entry
    entry.title = document.getElementById('field-title')?.value || '';
    entry.year = document.getElementById('field-year')?.value || '';
    entry.journal = document.getElementById('field-journal')?.value || '';
    entry.authors = (document.getElementById('field-authors')?.value || '').split('\n').filter(a => a.trim());
    entry.doi = document.getElementById('field-doi')?.value || '';
    entry.analysis_notes = document.getElementById('field-notes')?.value || '';
    entry.specialized_species = (document.getElementById('field-specialized-species')?.value || '')
        .split(',').map(s => s.trim()).filter(s => s);

    // Affiliations
    entry.affiliations = [];
    document.querySelectorAll('#affiliations-body tr').forEach(row => {
        const aff = {
            university: row.querySelector('.aff-university')?.value || '',
            country: row.querySelector('.aff-country')?.value || '',
            discipline: row.querySelector('.aff-discipline')?.value || ''
        };
        if (aff.university || aff.country || aff.discipline) {
            entry.affiliations.push(aff);
        }
    });
}

function debounceAutoSave(entry) {
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
        collectFormData(entry);
        saveEntry(entry.id, entry);
    }, 500);
}

function showSaveIndicator() {
    let indicator = document.querySelector('.saving-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'saving-indicator';
        indicator.textContent = '✓ Saved';
        document.body.appendChild(indicator);
    }
    indicator.classList.add('show');
    setTimeout(() => indicator.classList.remove('show'), 1500);
}

// ============================================================================
// Analytics Functions
// ============================================================================

// Get currently selected data sources from toggles
function getSelectedSources() {
    const sources = [];
    if (document.getElementById('toggle-human')?.checked) sources.push('human');
    if (document.getElementById('toggle-subset')?.checked) sources.push('subset');
    if (document.getElementById('toggle-fullset')?.checked) sources.push('fullset');
    // Return 'none' explicitly when no sources selected
    return sources.length > 0 ? sources.join(',') : 'none';
}

async function loadAnalytics() {
    const sources = getSelectedSources();
    const params = sources ? `?sources=${sources}` : '';
    return await api(`/analytics${params}`);
}

async function loadWordCloud(era) {
    const sources = getSelectedSources();
    const params = sources ? `?sources=${sources}` : '';
    return await api(`/analytics/wordcloud/${era}${params}`);
}

async function loadNetwork(type) {
    const sources = getSelectedSources();
    const params = sources ? `?sources=${sources}` : '';
    return await api(`/analytics/network/${type}${params}`);
}

// Refresh entries and analytics when source toggles change
function setupSourceToggleHandlers() {
    ['toggle-human', 'toggle-subset', 'toggle-fullset'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', async () => {
            // Reload entries with new source selection
            await loadEntries();

            // Also reload analytics if visible (check if analytics button is active)
            const analyticsBtn = document.getElementById('btn-analytics');
            const analyticsPanel = document.getElementById('analytics');
            const isAnalyticsVisible = analyticsBtn?.classList.contains('active') ||
                (analyticsPanel && analyticsPanel.style.display === 'block');

            if (isAnalyticsVisible) {
                // Show loading state
                const activeGroup = document.querySelector('.analytics-group.active, .analytics-group[style*="block"]');
                if (activeGroup) {
                    activeGroup.innerHTML = '<p class="loading-spinner">Refreshing analytics...</p>';
                }

                // Reload analytics data
                const data = await loadAnalytics();
                if (data) {
                    await renderAnalytics(data);
                }
            }
        });
    });
}


// Light theme for Plotly charts (optimized for downloading/printing)
// IMPORTANT: Use getPlotlyLayout() instead of spreading this directly,
// because Plotly mutates the layout object and corrupts shared xaxis/yaxis
const plotlyThemeBase = {
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#1f2328', size: 12 },
    autosize: true
};

// Factory function to get a FRESH layout object for each chart
// This prevents Plotly from mutating shared state between charts
function getPlotlyLayout(overrides = {}) {
    return {
        ...plotlyThemeBase,
        xaxis: {
            gridcolor: '#e1e4e8',
            linecolor: '#d0d7de',
            tickcolor: '#656d76',
            ...overrides.xaxis
        },
        yaxis: {
            gridcolor: '#e1e4e8',
            linecolor: '#d0d7de',
            tickcolor: '#656d76',
            ...overrides.yaxis
        },
        ...overrides
    };
}

// Plotly config with download enabled - saves PNG to your Downloads folder
const plotlyConfig = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToAdd: [],
    toImageButtonOptions: {
        format: 'png',
        filename: 'chart',
        height: 600,
        width: 1200,
        scale: 2
    }
};

// Color palette for charts
const chartColors = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7',
    '#79c0ff', '#56d364', '#e3b341', '#ff7b72', '#bc8cff',
    '#39d353', '#1f6feb', '#db6d28', '#da3633', '#8957e5'
];

async function renderAnalytics(data) {
    if (data.empty) {
        document.getElementById('group-longitudinal').innerHTML =
            '<p class="loading-spinner">No data yet. Commit some entries first.</p>';
        return;
    }

    // Render Group 1: Longitudinal Analysis
    renderVolumeChart(data);
    renderFeaturesEvolution(data);
    renderStagesEvolution(data);
    renderKeywordsChart(data);

    // Render Group 2: Distributions
    renderFeatureDistribution(data);
    renderStageDistribution(data);
    renderSpeciesCharts(data);
    renderDemographicsCharts(data);

    // Load word clouds (async)
    loadAndRenderWordClouds();

    // Load network graphs (async)
    loadAndRenderNetworks();
}

// ----- Group 1: Longitudinal Charts -----

function renderVolumeChart(data) {
    const years = Object.keys(data.papers_by_year || {}).map(Number).sort();
    const counts = years.map(y => data.papers_by_year[y] || 0);

    Plotly.newPlot('chart-volume', [{
        x: years,
        y: counts,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: chartColors[0], width: 2 },
        marker: { size: 8 },
        name: 'Papers'
    }], {
        title: { text: 'Number of Papers per Year', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 50 }
    }, plotlyConfig);
}

function renderFeaturesEvolution(data) {
    const years = Object.keys(data.features_by_year || {}).map(Number).sort();
    const features = data.all_features || [];

    const traces = features.map((feat, i) => ({
        x: years,
        y: years.map(y => (data.features_by_year[y] || {})[feat] || 0),
        type: 'scatter',
        mode: 'lines',
        stackgroup: 'one',
        name: feat.length > 25 ? feat.substring(0, 23) + '...' : feat,
        line: { color: chartColors[i % chartColors.length] }
    }));

    Plotly.newPlot('chart-features-evolution', traces, {
        title: { text: 'Linguistic Features Over Time (Stacked)', font: { size: 14 } },
        ...getPlotlyLayout(),
        showlegend: true,
        legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
        margin: { t: 40, r: 20, b: 100, l: 50 }
    }, plotlyConfig);
}

function renderStagesEvolution(data) {
    const years = Object.keys(data.stages_by_year || {}).map(Number).sort();
    const stages = data.all_stages || [];

    const traces = stages.map((stage, i) => ({
        x: years,
        y: years.map(y => (data.stages_by_year[y] || {})[stage] || 0),
        type: 'scatter',
        mode: 'lines',
        stackgroup: 'one',
        name: stage,
        line: { color: chartColors[i % chartColors.length] }
    }));

    Plotly.newPlot('chart-stages-evolution', traces, {
        title: { text: 'Computational Stages Over Time (Stacked)', font: { size: 14 } },
        ...getPlotlyLayout(),
        showlegend: true,
        legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
        margin: { t: 40, r: 20, b: 70, l: 50 }
    }, plotlyConfig);
}

function renderKeywordsChart(data) {
    const years = Object.keys(data.keywords_by_year || {}).map(Number).sort();
    const keywords = data.top_keywords || [];

    if (keywords.length === 0) {
        document.getElementById('chart-keywords').innerHTML = '<p class="loading-spinner">Not enough data for keyword analysis</p>';
        return;
    }

    const traces = keywords.map((kw, i) => ({
        x: years,
        y: years.map(y => (data.keywords_by_year[y] || {})[kw] || 0),
        type: 'scatter',
        mode: 'lines+markers',
        name: kw,
        line: { color: chartColors[i % chartColors.length], width: 2 },
        marker: { size: 6 }
    }));

    Plotly.newPlot('chart-keywords', traces, {
        title: { text: 'Top 5 Keywords Over Time', font: { size: 14 } },
        ...getPlotlyLayout(),
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        margin: { t: 40, r: 20, b: 60, l: 50 }
    }, plotlyConfig);
}

// ----- Group 2: Distribution Charts -----

function renderFeatureDistribution(data) {
    // Use actual data keys, sorted by count
    const featureEntries = Object.entries(data.feature_counts || {})
        .sort((a, b) => b[1] - a[1]);
    const features = featureEntries.map(e => e[0]);
    const counts = featureEntries.map(e => e[1]);

    Plotly.newPlot('chart-features-dist', [{
        y: features,
        x: counts,
        type: 'bar',
        orientation: 'h',
        marker: { color: chartColors[0] }
    }], {
        title: { text: 'Papers by Linguistic Feature', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 250 }
    }, plotlyConfig);
}

function renderStageDistribution(data) {
    // Use actual data keys, sorted by count
    const stageEntries = Object.entries(data.stage_counts || {})
        .sort((a, b) => b[1] - a[1]);
    const stages = stageEntries.map(e => e[0]);
    const counts = stageEntries.map(e => e[1]);

    Plotly.newPlot('chart-stages-dist', [{
        x: stages,
        y: counts,
        type: 'bar',
        marker: { color: chartColors.slice(0, stages.length) }
    }], {
        title: { text: 'Papers by Computational Stage', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 100, l: 50 },
        xaxis: { ...getPlotlyLayout().xaxis, tickangle: -30 }
    }, plotlyConfig);
}

function renderSpeciesCharts(data) {
    // Species categories - use actual data keys
    const catEntries = Object.entries(data.species_category_counts || {})
        .sort((a, b) => b[1] - a[1]);
    const categories = catEntries.map(e => e[0]);
    const catCounts = catEntries.map(e => e[1]);

    Plotly.newPlot('chart-species-cat', [{
        x: categories,
        y: catCounts,
        type: 'bar',
        marker: { color: chartColors.slice(0, categories.length) }
    }], {
        title: { text: 'Papers by Species Category', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 100, l: 50 },
        xaxis: { ...getPlotlyLayout().xaxis, tickangle: -30 }
    }, plotlyConfig);

    // Top specialized species
    const topSpecies = data.top_specialized_species || [];
    const speciesNames = topSpecies.map(s => s[0]);
    const speciesCounts = topSpecies.map(s => s[1]);

    Plotly.newPlot('chart-species-top', [{
        y: speciesNames.reverse(),
        x: speciesCounts.reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: chartColors[1] }
    }], {
        title: { text: 'Top 10 Specialized Species', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 150 }
    }, plotlyConfig);
}

function renderDemographicsCharts(data) {
    // Countries
    const countries = Object.entries(data.country_counts || {})
        .sort((a, b) => b[1] - a[1]).slice(0, 10);

    Plotly.newPlot('chart-countries', [{
        y: countries.map(c => c[0]).reverse(),
        x: countries.map(c => c[1]).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: chartColors[0] }
    }], {
        title: { text: 'Papers by Country (Top 10)', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 100 }
    }, plotlyConfig);

    // Disciplines
    const disciplines = Object.entries(data.discipline_counts || {})
        .sort((a, b) => b[1] - a[1]);

    Plotly.newPlot('chart-disciplines', [{
        labels: disciplines.map(d => d[0]),
        values: disciplines.map(d => d[1]),
        type: 'pie',
        marker: { colors: chartColors },
        textinfo: 'label+percent',
        textposition: 'inside'
    }], {
        title: { text: 'Papers by Discipline', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 20 },
        showlegend: false
    }, plotlyConfig);

    // Affiliations
    const affiliations = data.top_affiliations || [];

    Plotly.newPlot('chart-affiliations', [{
        y: affiliations.map(a => a[0]).slice(0, 10).reverse(),
        x: affiliations.map(a => a[1]).slice(0, 10).reverse(),
        type: 'bar',
        orientation: 'h',
        marker: { color: chartColors[2] }
    }], {
        title: { text: 'Top 10 Affiliations', font: { size: 14 } },
        ...getPlotlyLayout(),
        margin: { t: 40, r: 20, b: 40, l: 180 }
    }, plotlyConfig);
}

// ----- Group 3: Word Clouds -----

async function loadAndRenderWordClouds() {
    // Pre-LLM
    document.getElementById('wordcloud-pre').innerHTML = '<p class="loading">Loading...</p>';
    try {
        const preData = await loadWordCloud('pre');
        if (preData.image) {
            document.getElementById('wordcloud-pre').innerHTML =
                `<img src="${preData.image}" alt="Pre-LLM Word Cloud"><p style="font-size:0.75rem;color:var(--text-muted)">${preData.paper_count} papers</p>`;
        } else if (preData.error) {
            document.getElementById('wordcloud-pre').innerHTML = `<p class="error-message">${preData.error}</p>`;
        }
    } catch (e) {
        document.getElementById('wordcloud-pre').innerHTML = '<p class="error-message">Failed to load</p>';
    }

    // Post-LLM
    document.getElementById('wordcloud-post').innerHTML = '<p class="loading">Loading...</p>';
    try {
        const postData = await loadWordCloud('post');
        if (postData.image) {
            document.getElementById('wordcloud-post').innerHTML =
                `<img src="${postData.image}" alt="Post-LLM Word Cloud"><p style="font-size:0.75rem;color:var(--text-muted)">${postData.paper_count} papers</p>`;
        } else if (postData.error) {
            document.getElementById('wordcloud-post').innerHTML = `<p class="error-message">${postData.error}</p>`;
        }
    } catch (e) {
        document.getElementById('wordcloud-post').innerHTML = '<p class="error-message">Failed to load</p>';
    }
}

// ----- Group 4: Network Graphs -----

async function loadAndRenderNetworks() {
    const networks = ['affiliation', 'country', 'discipline'];

    for (const type of networks) {
        const chartEl = document.getElementById(`network-${type}`);
        const statsEl = document.getElementById(`network-${type}-stats`);

        chartEl.innerHTML = '<p class="loading">Loading...</p>';

        try {
            const data = await loadNetwork(type);

            if (data.error) {
                chartEl.innerHTML = `<p class="error-message">${data.error}</p>`;
                continue;
            }

            if (data.nodes.length === 0) {
                chartEl.innerHTML = '<p class="loading-spinner">No network data available</p>';
                continue;
            }

            // Create network visualization using Plotly scatter
            renderNetworkGraph(chartEl.id, data);

            // Show stats
            statsEl.innerHTML = `
                <span class="stat-item"><span class="stat-label">Nodes:</span> <span class="stat-value">${data.node_count}</span></span>
                <span class="stat-item"><span class="stat-label">Edges:</span> <span class="stat-value">${data.edge_count}</span></span>
            `;
        } catch (e) {
            chartEl.innerHTML = '<p class="error-message">Failed to load network</p>';
        }
    }
}

function renderNetworkGraph(containerId, data) {
    // Simple force-directed layout approximation using Plotly
    const nodes = data.nodes || [];
    const edges = data.edges || [];

    // Create positions using simple circular layout
    const nodePositions = {};
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        const radius = 1 + Math.log(node.size + 1) * 0.3;
        nodePositions[node.id] = {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius
        };
    });

    // Edge traces
    const edgeX = [];
    const edgeY = [];
    edges.forEach(edge => {
        const source = nodePositions[edge.source];
        const target = nodePositions[edge.target];
        if (source && target) {
            edgeX.push(source.x, target.x, null);
            edgeY.push(source.y, target.y, null);
        }
    });

    const edgeTrace = {
        x: edgeX,
        y: edgeY,
        mode: 'lines',
        line: { width: 0.5, color: '#30363d' },
        hoverinfo: 'none',
        type: 'scatter'
    };

    // Node trace
    const nodeTrace = {
        x: nodes.map(n => nodePositions[n.id].x),
        y: nodes.map(n => nodePositions[n.id].y),
        mode: 'markers+text',
        marker: {
            size: nodes.map(n => 8 + Math.min(n.size * 3, 25)),
            color: chartColors[0],
            line: { width: 1, color: '#161b22' }
        },
        text: nodes.map(n => n.label.length > 15 ? n.label.substring(0, 13) + '...' : n.label),
        textposition: 'top center',
        textfont: { size: 8, color: '#8b949e' },
        hovertext: nodes.map(n => `${n.label} (${n.size} papers)`),
        hoverinfo: 'text',
        type: 'scatter'
    };

    Plotly.newPlot(containerId, [edgeTrace, nodeTrace], {
        ...getPlotlyLayout(),
        showlegend: false,
        hovermode: 'closest',
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        margin: { t: 10, r: 10, b: 10, l: 10 }
    }, plotlyConfig);
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// ============================================================================
// Event Handlers
// ============================================================================

function setupEventHandlers() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.activeTab = tab.dataset.tab;
            state.selectedEntryId = null;
            renderEntryList();
            renderEditor();
        });
    });

    // Import modal
    document.getElementById('btn-import').addEventListener('click', () => {
        document.getElementById('import-modal').style.display = 'flex';
    });

    document.getElementById('modal-close').addEventListener('click', () => {
        document.getElementById('import-modal').style.display = 'none';
    });

    document.getElementById('btn-parse').addEventListener('click', async () => {
        const bibtex = document.getElementById('bibtex-input').value;
        if (!bibtex) {
            alert('Paste BibTeX content first');
            return;
        }
        const result = await importBibtex(bibtex);
        if (result.error) {
            alert('Error: ' + result.error);
        } else {
            document.getElementById('import-modal').style.display = 'none';
            document.getElementById('bibtex-input').value = '';
        }
    });

    // Analytics toggle
    document.getElementById('btn-analytics').addEventListener('click', async () => {
        const editor = document.getElementById('editor');
        const analytics = document.getElementById('analytics');

        if (analytics.style.display === 'none') {
            editor.style.display = 'none';
            analytics.style.display = 'block';
            document.getElementById('btn-analytics').classList.add('active');

            // Load analytics data
            const data = await loadAnalytics();
            renderAnalytics(data);
        } else {
            editor.style.display = 'block';
            analytics.style.display = 'none';
            document.getElementById('btn-analytics').classList.remove('active');
        }
    });

    // Analytics tab switching
    document.querySelectorAll('.analytics-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Update tab styles
            document.querySelectorAll('.analytics-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show corresponding group
            const group = tab.dataset.group;
            document.querySelectorAll('.analytics-group').forEach(g => {
                g.classList.remove('active');
                g.style.display = 'none';
            });
            const targetGroup = document.getElementById(`group-${group}`);
            if (targetGroup) {
                targetGroup.classList.add('active');
                targetGroup.style.display = 'block';
            }
        });
    });

    // Close modal on outside click
    document.getElementById('import-modal').addEventListener('click', (e) => {
        if (e.target.id === 'import-modal') {
            document.getElementById('import-modal').style.display = 'none';
        }
    });
}

// ============================================================================
// Theme Toggle
// ============================================================================

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeButton(newTheme);

    // Update Plotly charts if visible
    updatePlotlyTheme(newTheme);
}

function updateThemeButton(theme) {
    const btn = document.getElementById('btn-theme');
    if (btn) {
        btn.textContent = theme === 'dark' ? '☀️' : '🌙';
        btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
    }
}

function updatePlotlyTheme(theme) {
    const bgColor = theme === 'dark' ? '#161b22' : '#f6f8fa';
    const textColor = theme === 'dark' ? '#e6edf3' : '#1f2328';
    const gridColor = theme === 'dark' ? '#30363d' : '#d0d7de';

    // Update all Plotly charts on the page
    const plotlyCharts = document.querySelectorAll('.js-plotly-plot');
    plotlyCharts.forEach(chart => {
        Plotly.relayout(chart, {
            paper_bgcolor: bgColor,
            plot_bgcolor: bgColor,
            font: { color: textColor },
            'xaxis.gridcolor': gridColor,
            'yaxis.gridcolor': gridColor
        });
    });
}

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventHandlers();
    setupSourceToggleHandlers();
    loadEntries();

    // Theme toggle listener
    document.getElementById('btn-theme')?.addEventListener('click', toggleTheme);
});

