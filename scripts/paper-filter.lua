-- Structure Paper.md into a clean academic PDF without changing wording.
local stringify = pandoc.utils.stringify

local UNNUMBERED = {
  ["Declaration on the use of artificial intelligence"] = true,
  ["References"] = true,
}

local function strip_heading_number(text)
  local s = text
  s = s:gsub("^%d+%.%d+%.%d+%s+", "")
  s = s:gsub("^%d+%.%d+%s+", "")
  s = s:gsub("^%d+%.%s+", "")
  return s
end

local function is_abstract(block)
  if block.t ~= "BlockQuote" then
    return false
  end
  local t = stringify(block)
  return t:match("^%s*Abstract") ~= nil
end

local function extract_abstract(blockquote)
  local inlines = {}
  for _, block in ipairs(blockquote.content) do
    if block.t == "Para" or block.t == "Plain" then
      for _, inl in ipairs(block.content) do
        table.insert(inlines, inl)
      end
    end
  end
  local start = 1
  if inlines[1] and inlines[1].t == "Strong" then
    local t = stringify(inlines[1]):gsub("%.$", "")
    if t:lower() == "abstract" then
      start = 2
      while inlines[start] and (
          inlines[start].t == "Space"
          or inlines[start].t == "SoftBreak"
          or inlines[start].t == "LineBreak"
        ) do
        start = start + 1
      end
    end
  end
  local rest = {}
  for i = start, #inlines do
    table.insert(rest, inlines[i])
  end
  return pandoc.MetaBlocks({pandoc.Para(rest)})
end

local function skip_breaks(inlines, j)
  while j <= #inlines and (
      inlines[j].t == "SoftBreak"
      or inlines[j].t == "LineBreak"
      or inlines[j].t == "Space"
    ) do
    j = j + 1
  end
  return j
end

local function is_solo_image_para(block)
  return block.t == "Para"
    and #block.content == 1
    and block.content[1].t == "Image"
end

local function is_figure_block(block)
  return block.t == "Figure" or is_solo_image_para(block)
end

-- Paper.md often puts the image and "**Figure N.** ..." in the same paragraph
-- (no blank line). Split those so they become a real figure float.
local function split_image_caption_para(block)
  if block.t ~= "Para" then
    return nil
  end
  local content = block.content
  if #content < 2 or content[1].t ~= "Image" then
    return nil
  end
  local j = skip_breaks(content, 2)
  if j > #content then
    return nil
  end
  local rest = {}
  for k = j, #content do
    table.insert(rest, content[k])
  end
  local s = stringify(pandoc.Span(rest)):gsub("^%s+", "")
  if not s:match("^Figure%s+%d+") then
    return nil
  end
  return content[1], rest
end

local function starts_with_figure_label(block)
  if block.t ~= "Para" then
    return false
  end
  local s = stringify(block):gsub("^%s+", "")
  return s:match("^Figure%s+%d+") ~= nil
end

local function is_caption_continuation(block)
  if block.t ~= "Para" then
    return false
  end
  local s = stringify(block):gsub("^%s+", "")
  -- Only panel-label fragments, never ordinary sentences.
  if s:match("^%([a-z]%)") then return true end
  if s:match("^Bottom:") or s:match("^Top:") then return true end
  local token = s:match("^(%S+)")
  if token and #token <= 10 and token:match(":$") and token:match("^%l") then
    return true
  end
  return false
end

local function unwrap_caption_inlines(inlines)
  if #inlines == 1 and inlines[1].t == "Emph" then
    inlines = inlines[1].content
  end
  -- Drop a leftover closing italic asterisk from "**Figure N.** ...*"
  if #inlines > 0 then
    local last = inlines[#inlines]
    if last.t == "Str" then
      last.text = last.text:gsub("%*$", "")
    end
  end
  return inlines
end

local function merge_caption_paras(paras)
  local inlines = {}
  for i, para in ipairs(paras) do
    local content = unwrap_caption_inlines(para.content)
    if i > 1 then
      table.insert(inlines, pandoc.Space())
    end
    for _, inl in ipairs(content) do
      table.insert(inlines, inl)
    end
  end
  return inlines
end

local function set_figure_caption(block, cap_inlines)
  -- Convert leftover markdown line-breaks so the caption wraps as prose.
  local cleaned = {}
  for _, inl in ipairs(cap_inlines) do
    if inl.t == "SoftBreak" or inl.t == "LineBreak" then
      table.insert(cleaned, pandoc.Space())
    else
      table.insert(cleaned, inl)
    end
  end
  local caption = pandoc.Caption({pandoc.Plain(cleaned)}, cleaned)
  local function with_here(fig)
    fig.attr.attributes["pos"] = "H"
    return fig
  end
  if block.t == "Figure" then
    block.caption = caption
    return with_here(block)
  end
  local img = block.content[1]
  img.caption = cleaned
  return with_here(pandoc.Figure({pandoc.Plain({img})}, caption))
end

local function process_heading(header)
  header.level = header.level - 1
  if header.level < 1 then
    header.level = 1
  end
  local title = strip_heading_number(stringify(header.content))
  header.content = {pandoc.Str(title)}
  if UNNUMBERED[title] then
    header.classes:insert("unnumbered")
  end
  return header
end

local function collect_caption_paras(blocks, i)
  local paras = {blocks[i]}
  i = i + 1
  while i <= #blocks and is_caption_continuation(blocks[i]) do
    table.insert(paras, blocks[i])
    i = i + 1
  end
  return paras, i
end

function Code(el)
  if #el.text > 24 then
    local escaped = el.text:gsub("\\", "\\textbackslash{}"):gsub("_", "\\_"):gsub("%%", "\\%%")
    return pandoc.RawInline("latex", "\\texttt{\\seqsplit{" .. escaped .. "}}")
  end
end

function CodeBlock(el)
  el.text = el.text:gsub("↑", "^")
  return el
end

function Pandoc(doc)
  local meta = doc.meta
  local src = doc.blocks
  local out = {}
  local i = 1
  local got_title = false
  local got_authors = false

  while i <= #src do
    local b = src[i]

    if b.t == "Header" and b.level == 1 and not got_title then
      meta.title = pandoc.MetaInlines(b.content)
      got_title = true
      i = i + 1

    elseif (not got_authors) and got_title and b.t == "Para" then
      local t = stringify(b)
      if t:match("Moldwin") or t:match("Toviah") or t:match("Mahajne") then
        local names, aff = {}, {}
        local seen_break = false
        for _, inl in ipairs(b.content) do
          if inl.t == "LineBreak" or inl.t == "SoftBreak" then
            seen_break = true
          elseif not seen_break then
            table.insert(names, inl)
          else
            table.insert(aff, inl)
          end
        end
        if src[i + 1] and src[i + 1].t == "Para" then
          local t2 = stringify(src[i + 1])
          if t2:match("Safra") or t2:match("Hebrew") or t2:match("Edmond") then
            aff = src[i + 1].content
            i = i + 1
          end
        end
        local author = {}
        for _, inl in ipairs(names) do
          table.insert(author, inl)
        end
        if #aff > 0 then
          table.insert(author, pandoc.RawInline("latex", "\\\\[0.45em]{\\normalsize\\itshape "))
          for _, inl in ipairs(aff) do
            table.insert(author, inl)
          end
          table.insert(author, pandoc.RawInline("latex", "}"))
        end
        meta.author = pandoc.MetaInlines(author)
        got_authors = true
        i = i + 1
      else
        table.insert(out, b)
        i = i + 1
      end

    elseif is_abstract(b) then
      meta.abstract = extract_abstract(b)
      i = i + 1

    elseif b.t == "HorizontalRule" then
      i = i + 1

    elseif split_image_caption_para(b) then
      local img, rest = split_image_caption_para(b)
      local cap_paras = {pandoc.Para(rest)}
      i = i + 1
      while i <= #src and is_caption_continuation(src[i]) do
        table.insert(cap_paras, src[i])
        i = i + 1
      end
      local cap = merge_caption_paras(cap_paras)
      table.insert(out, set_figure_caption(pandoc.Para({img}), cap))

    elseif is_figure_block(b) and src[i + 1] and starts_with_figure_label(src[i + 1]) then
      local paras
      paras, i = collect_caption_paras(src, i + 1)
      local cap = merge_caption_paras(paras)
      table.insert(out, set_figure_caption(b, cap))

    elseif b.t == "Header" then
      table.insert(out, process_heading(b))
      i = i + 1

    else
      table.insert(out, b)
      i = i + 1
    end
  end

  -- Hanging-indent references: style the list after the References heading.
  for idx, block in ipairs(out) do
    if block.t == "Header" and stringify(block.content) == "References" then
      if out[idx + 1] and out[idx + 1].t == "BulletList" then
        table.insert(out, idx + 1, pandoc.RawBlock(
          "latex",
          "\\begingroup\\raggedright\\setlist[itemize]{label={},leftmargin=1.65em,itemsep=0.42em,topsep=0.5em,parsep=0.12em}"
        ))
        table.insert(out, idx + 3, pandoc.RawBlock("latex", "\\endgroup"))
      end
      break
    end
  end

  meta.date = pandoc.MetaString("")
  return pandoc.Pandoc(out, meta):walk({
    Figure = function(fig)
      fig.attr.attributes["pos"] = "H"
      return fig
    end
  })
end
